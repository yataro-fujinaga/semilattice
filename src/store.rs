use anyhow::Result;
use rusqlite::Connection;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::embedding;

pub struct Store {
    conn: Connection,
}

/// A file result with its relevance score and the traversal path that found it.
#[derive(Debug)]
pub struct FileResult {
    pub path: String,
    pub score: f32,
    pub trails: Vec<Trail>,
}

/// How a file was reached: through which context, at what hop distance, with what score.
#[derive(Debug, Clone)]
pub struct Trail {
    pub context_label: String,
    pub hop: u32,
    pub score: f32,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        Ok(Self { conn })
    }

    pub fn init(&self) -> Result<()> {
        crate::db::create_tables(&self.conn)?;
        Ok(())
    }

    pub fn add_context(
        &self,
        label: &str,
        files: &[String],
        emb: Option<&[f32]>,
    ) -> Result<usize> {
        self.init()?;

        let emb_blob = emb.map(embedding::vec_to_bytes);
        self.conn.execute(
            "INSERT INTO contexts (label, embedding) VALUES (?1, ?2)",
            rusqlite::params![label, emb_blob],
        )?;
        let context_id = self.conn.last_insert_rowid();

        for file in files {
            self.conn.execute(
                "INSERT INTO files (path) VALUES (?1) ON CONFLICT(path) DO NOTHING",
                rusqlite::params![file],
            )?;
            let file_id: i64 = self.conn.query_row(
                "SELECT id FROM files WHERE path = ?1",
                rusqlite::params![file],
                |row| row.get(0),
            )?;
            self.conn.execute(
                "INSERT OR IGNORE INTO context_files (context_id, file_id) VALUES (?1, ?2)",
                rusqlite::params![context_id, file_id],
            )?;
        }

        Ok(files.len())
    }

    /// Graph-based semantic query.
    ///
    /// 1. Find entry contexts via embedding similarity (top-k)
    /// 2. Get files from entry contexts
    /// 3. Find neighboring contexts that share files (1 hop)
    /// 4. Get files from neighboring contexts (2 hops)
    /// 5. Score decays with each hop
    pub fn query_graph(
        &self,
        query_emb: &[f32],
        top_k: usize,
        max_hops: u32,
        decay: f32,
    ) -> Result<Vec<FileResult>> {
        self.init()?;

        // Step 1: Find entry contexts by embedding similarity
        let entry_contexts = self.find_top_contexts(query_emb, top_k)?;

        if entry_contexts.is_empty() {
            return Ok(vec![]);
        }

        // Track visited contexts and file scores
        let mut visited_contexts: HashSet<i64> = HashSet::new();
        let mut file_scores: HashMap<String, (f32, Vec<Trail>)> = HashMap::new();

        // BFS frontier: (context_id, context_label, score, hop)
        let mut frontier: Vec<(i64, String, f32, u32)> = entry_contexts
            .iter()
            .map(|(id, label, score)| (*id, label.clone(), *score, 0))
            .collect();

        for hop in 0..=max_hops {
            let mut next_frontier: Vec<(i64, String, f32, u32)> = Vec::new();

            for (ctx_id, ctx_label, ctx_score, _) in &frontier {
                if !visited_contexts.insert(*ctx_id) {
                    continue;
                }

                let hop_score = ctx_score * decay.powi(hop as i32);

                // Get files for this context
                let files = self.get_context_files(*ctx_id)?;

                for file_path in &files {
                    let entry = file_scores
                        .entry(file_path.clone())
                        .or_insert((0.0, Vec::new()));
                    // Keep the best score
                    if hop_score > entry.0 {
                        entry.0 = hop_score;
                    }
                    entry.1.push(Trail {
                        context_label: ctx_label.clone(),
                        hop,
                        score: hop_score,
                    });
                }

                // Find neighboring contexts via shared files (for next hop)
                if hop < max_hops {
                    let neighbors = self.find_neighbors(*ctx_id, &visited_contexts)?;
                    for (neighbor_id, neighbor_label, shared_count) in neighbors {
                        // Neighbor score: parent score * decay * (shared files as confidence)
                        let neighbor_score =
                            ctx_score * (shared_count as f32).min(3.0) / 3.0;
                        next_frontier.push((
                            neighbor_id,
                            neighbor_label,
                            neighbor_score,
                            hop + 1,
                        ));
                    }
                }
            }

            frontier = next_frontier;
        }

        // Collect and sort results
        let mut results: Vec<FileResult> = file_scores
            .into_iter()
            .map(|(path, (score, trails))| FileResult {
                path,
                score,
                trails,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Find top-k contexts by embedding similarity.
    fn find_top_contexts(
        &self,
        query_emb: &[f32],
        top_k: usize,
    ) -> Result<Vec<(i64, String, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.label, c.embedding
             FROM contexts c
             WHERE c.embedding IS NOT NULL",
        )?;

        let mut scored: Vec<(i64, String, f32)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let label: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                Ok((id, label, blob))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, label, blob)| {
                let emb = embedding::bytes_to_vec(&blob);
                let score = embedding::cosine_similarity(query_emb, &emb);
                (id, label, score)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored)
    }

    /// Get file paths for a context.
    fn get_context_files(&self, context_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path FROM files f
             JOIN context_files cf ON cf.file_id = f.id
             WHERE cf.context_id = ?1
             ORDER BY f.path",
        )?;
        let files = stmt
            .query_map(rusqlite::params![context_id], |row| {
                row.get::<_, String>(0)
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(files)
    }

    /// Find contexts that share files with the given context, excluding already visited ones.
    /// Returns (context_id, label, shared_file_count).
    fn find_neighbors(
        &self,
        context_id: i64,
        visited: &HashSet<i64>,
    ) -> Result<Vec<(i64, String, usize)>> {
        // Find contexts sharing at least one file
        let mut stmt = self.conn.prepare(
            "SELECT c2.id, c2.label, COUNT(*) as shared
             FROM context_files cf1
             JOIN context_files cf2 ON cf2.file_id = cf1.file_id AND cf2.context_id != cf1.context_id
             JOIN contexts c2 ON c2.id = cf2.context_id
             WHERE cf1.context_id = ?1
             GROUP BY c2.id
             ORDER BY shared DESC",
        )?;

        let neighbors: Vec<(i64, String, usize)> = stmt
            .query_map(rusqlite::params![context_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, usize>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .filter(|(id, _, _)| !visited.contains(id))
            .collect();

        Ok(neighbors)
    }

    /// Text-based query (fallback).
    pub fn query_text(&self, term: &str) -> Result<Vec<(String, String)>> {
        self.init()?;

        let words: Vec<&str> = term.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        let conditions: Vec<String> = words
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c.label LIKE ?{}", i + 1))
            .collect();
        let where_clause = conditions.join(" OR ");

        let sql = format!(
            "SELECT DISTINCT c.label, f.path
             FROM contexts c
             JOIN context_files cf ON cf.context_id = c.id
             JOIN files f ON f.id = cf.file_id
             WHERE {}
             ORDER BY c.created_at DESC, f.path",
            where_clause
        );

        let params: Vec<String> = words.iter().map(|w| format!("%{}%", w)).collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
            .iter()
            .map(|p| p as &dyn rusqlite::types::ToSql)
            .collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn list_contexts(&self) -> Result<Vec<(String, i64, String)>> {
        self.init()?;

        let mut stmt = self.conn.prepare(
            "SELECT c.label, COUNT(cf.file_id), c.created_at
             FROM contexts c
             LEFT JOIN context_files cf ON cf.context_id = c.id
             GROUP BY c.id
             ORDER BY c.created_at DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn show_context(&self, context: &str) -> Result<Vec<(String, Vec<String>)>> {
        self.init()?;

        let pattern = format!("%{}%", context);
        let mut stmt = self.conn.prepare(
            "SELECT c.label, f.path
             FROM contexts c
             JOIN context_files cf ON cf.context_id = c.id
             JOIN files f ON f.id = cf.file_id
             WHERE c.label LIKE ?1
             ORDER BY c.label, f.path",
        )?;

        let rows = stmt.query_map(rusqlite::params![pattern], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for row in rows {
            let (label, path) = row?;
            grouped.entry(label).or_default().push(path);
        }

        Ok(grouped.into_iter().collect())
    }
}
