use std::collections::HashMap;
use std::path::Path;

use rusqlite::{params, Connection};

use crate::state::{Experiment, ExperimentConfig, SteeringVectorSet};

pub fn init_db(path: &Path) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS steering_vectors (
            name TEXT PRIMARY KEY,
            concept TEXT NOT NULL,
            vectors_json TEXT NOT NULL,
            num_training_pairs INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            config_json TEXT NOT NULL,
            results_json TEXT
        );",
    )?;
    Ok(())
}

pub fn save_steering_vector(path: &Path, svec: &SteeringVectorSet) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    let vectors_json = serde_json::to_string(&svec.vectors).unwrap_or_default();
    conn.execute(
        "INSERT OR REPLACE INTO steering_vectors
         (name, concept, vectors_json, num_training_pairs, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            svec.name,
            svec.concept,
            vectors_json,
            svec.num_training_pairs as i64,
            svec.created_at
        ],
    )?;
    Ok(())
}

pub fn load_steering_vectors(
    path: &Path,
) -> rusqlite::Result<HashMap<String, SteeringVectorSet>> {
    let conn = Connection::open(path)?;
    let mut stmt = conn.prepare(
        "SELECT name, concept, vectors_json, num_training_pairs, created_at
         FROM steering_vectors",
    )?;
    let mut map = HashMap::new();
    let rows = stmt.query_map([], |row| {
        let name: String = row.get(0)?;
        let concept: String = row.get(1)?;
        let vectors_json: String = row.get(2)?;
        let num_training_pairs: i64 = row.get(3)?;
        let created_at: String = row.get(4)?;
        let vectors: HashMap<usize, Vec<f32>> =
            serde_json::from_str(&vectors_json).unwrap_or_default();
        Ok(SteeringVectorSet {
            name,
            concept,
            vectors,
            num_training_pairs: num_training_pairs as usize,
            created_at,
        })
    })?;
    for row in rows {
        let svec = row?;
        map.insert(svec.name.clone(), svec);
    }
    Ok(map)
}

pub fn save_experiment(path: &Path, exp: &Experiment) -> rusqlite::Result<()> {
    let conn = Connection::open(path)?;
    let config_json = serde_json::to_string(&exp.config).unwrap_or_default();
    let results_json = exp
        .results
        .as_ref()
        .map(|r| serde_json::to_string(r).unwrap_or_default());
    conn.execute(
        "INSERT OR REPLACE INTO experiments
         (id, name, status, created_at, config_json, results_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            exp.id,
            exp.name,
            exp.status,
            exp.created_at,
            config_json,
            results_json
        ],
    )?;
    Ok(())
}

pub fn load_experiments(path: &Path) -> rusqlite::Result<HashMap<String, Experiment>> {
    let conn = Connection::open(path)?;
    let mut stmt = conn.prepare(
        "SELECT id, name, status, created_at, config_json, results_json FROM experiments",
    )?;
    let mut map = HashMap::new();
    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let config_json: String = row.get(4)?;
        let results_json: Option<String> = row.get(5)?;
        Ok(Experiment {
            id,
            name: row.get(1)?,
            status: row.get(2)?,
            created_at: row.get(3)?,
            config: serde_json::from_str(&config_json).unwrap_or(ExperimentConfig {
                experiment_type: "unknown".into(),
                prompt: String::new(),
                steering_layers: None,
                steering_scale: None,
            }),
            results: results_json.and_then(|j| serde_json::from_str(&j).ok()),
        })
    })?;
    for row in rows {
        if let Ok(exp) = row {
            map.insert(exp.id.clone(), exp);
        }
    }
    Ok(map)
}
