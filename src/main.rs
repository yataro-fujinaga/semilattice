mod db;
mod embedding;
mod store;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sl", about = "Semilattice — semantic relationship layer for your files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize .sl/ in the current directory
    Init,

    /// Record a relationship: sl add "context label" file1 file2 ...
    Add {
        /// The context label (e.g. "tax calculation bugfix")
        context: String,
        /// Files related to this context
        files: Vec<PathBuf>,
    },

    /// Query for related files: sl query "search term"
    Query {
        /// The search term
        term: String,

        /// Use text matching only (skip semantic search)
        #[arg(long)]
        text_only: bool,

        /// Number of entry contexts to find via embedding (default 3)
        #[arg(long, default_value = "3")]
        top_k: usize,

        /// Max hops to traverse from entry contexts (default 2)
        #[arg(long, default_value = "2")]
        max_hops: u32,

        /// Score decay factor per hop (default 0.7)
        #[arg(long, default_value = "0.7")]
        decay: f32,
    },

    /// List all recorded contexts
    Contexts,

    /// Show files in a specific context
    Show {
        /// Context label or partial match
        context: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add { context, files } => cmd_add(&context, &files),
        Commands::Query {
            term,
            text_only,
            top_k,
            max_hops,
            decay,
        } => cmd_query(&term, text_only, top_k, max_hops, decay),
        Commands::Contexts => cmd_contexts(),
        Commands::Show { context } => cmd_show(&context),
    }
}

fn cmd_init() -> Result<()> {
    let sl_dir = PathBuf::from(".sl");
    if sl_dir.exists() {
        println!(".sl/ already exists");
        return Ok(());
    }
    std::fs::create_dir(&sl_dir)?;
    let s = store::Store::open(sl_dir.join("relations.db"))?;
    s.init()?;
    println!("Initialized .sl/ in current directory");
    Ok(())
}

fn find_sl_dir() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        let sl = dir.join(".sl");
        if sl.is_dir() {
            return Ok(sl);
        }
        if !dir.pop() {
            anyhow::bail!(
                "Not in a semilattice repository (no .sl/ found). Run `sl init` first."
            );
        }
    }
}

fn cmd_add(context: &str, files: &[PathBuf]) -> Result<()> {
    if files.is_empty() {
        anyhow::bail!("No files specified. Usage: sl add \"context\" file1 file2 ...");
    }

    let sl_dir = find_sl_dir()?;
    let s = store::Store::open(sl_dir.join("relations.db"))?;

    // Embed the context label
    let embedder = embedding::Embedder::new(sl_dir.join("models"))?;
    let emb = embedder.embed_context(context)?;

    let repo_root = sl_dir.parent().unwrap();
    let cwd = std::env::current_dir()?;

    let mut resolved: Vec<String> = Vec::new();
    for f in files {
        let abs = if f.is_absolute() {
            f.clone()
        } else {
            cwd.join(f)
        };
        let abs = abs.canonicalize().unwrap_or(abs);
        let rel = abs
            .strip_prefix(repo_root)
            .unwrap_or(&abs)
            .to_string_lossy()
            .to_string();
        resolved.push(rel);
    }

    let count = s.add_context(context, &resolved, Some(&emb))?;
    println!("[{}] {} files recorded", context, count);
    for f in &resolved {
        println!("  {}", f);
    }
    Ok(())
}

fn cmd_query(term: &str, text_only: bool, top_k: usize, max_hops: u32, decay: f32) -> Result<()> {
    let sl_dir = find_sl_dir()?;
    let s = store::Store::open(sl_dir.join("relations.db"))?;

    if text_only {
        return cmd_query_text(&s, term);
    }

    // Graph-based semantic search
    let embedder = embedding::Embedder::new(sl_dir.join("models"))?;
    let query_emb = embedder.embed_query(term)?;

    let results = s.query_graph(&query_emb, top_k, max_hops, decay)?;

    if results.is_empty() {
        println!("(no semantic matches, falling back to text search)");
        return cmd_query_text(&s, term);
    }

    let ctx_count = {
        let mut s: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for r in &results {
            for t in &r.trails {
                s.insert(t.context_label.clone());
            }
        }
        s.len()
    };

    println!(
        "\"{}\" → {} files (via {} contexts)",
        term,
        results.len(),
        ctx_count,
    );
    println!();

    for r in &results {
        println!("  {} (score: {:.2})", r.path, r.score);
        for t in &r.trails {
            let hop_label = if t.hop == 0 {
                "direct".to_string()
            } else {
                format!("{}hop", t.hop)
            };
            println!(
                "    via [{}] ({}, {:.2})",
                t.context_label, hop_label, t.score
            );
        }
    }

    Ok(())
}

fn cmd_query_text(s: &store::Store, term: &str) -> Result<()> {
    let results = s.query_text(term)?;

    if results.is_empty() {
        println!("No files found for \"{}\"", term);
        return Ok(());
    }

    let mut file_contexts: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for (context, file) in &results {
        file_contexts
            .entry(file.clone())
            .or_default()
            .push(context.clone());
    }

    let ctx_count = {
        let mut set: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (ctx, _) in &results {
            set.insert(ctx);
        }
        set.len()
    };

    println!(
        "\"{}\" → {} files (from {} contexts, text match)",
        term,
        file_contexts.len(),
        ctx_count,
    );
    println!();

    for (file, contexts) in &file_contexts {
        println!("  {}", file);
        for ctx in contexts {
            println!("    via [{}]", ctx);
        }
    }

    Ok(())
}

fn cmd_contexts() -> Result<()> {
    let sl_dir = find_sl_dir()?;
    let s = store::Store::open(sl_dir.join("relations.db"))?;

    let contexts = s.list_contexts()?;

    if contexts.is_empty() {
        println!("No contexts recorded yet.");
        return Ok(());
    }

    for (label, file_count, created) in &contexts {
        println!("[{}] {} files ({})", label, file_count, created);
    }

    Ok(())
}

fn cmd_show(context: &str) -> Result<()> {
    let sl_dir = find_sl_dir()?;
    let s = store::Store::open(sl_dir.join("relations.db"))?;

    let files = s.show_context(context)?;

    if files.is_empty() {
        println!("No context matching \"{}\"", context);
        return Ok(());
    }

    for (ctx_label, file_list) in &files {
        println!("[{}]", ctx_label);
        for f in file_list {
            println!("  {}", f);
        }
    }

    Ok(())
}
