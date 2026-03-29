use anyhow::Result;
use rusqlite::Connection;

pub fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS contexts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label       TEXT NOT NULL,
            embedding   BLOB,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS context_files (
            context_id  INTEGER NOT NULL REFERENCES contexts(id),
            file_id     INTEGER NOT NULL REFERENCES files(id),
            PRIMARY KEY (context_id, file_id)
        );

        CREATE INDEX IF NOT EXISTS idx_contexts_label ON contexts(label);
        CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
        ",
    )?;
    Ok(())
}
