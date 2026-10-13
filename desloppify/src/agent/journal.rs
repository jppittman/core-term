//! [`Ask`](super::Ask) that remembers: every answer is appended to a JSON
//! Lines file as it arrives, and a question already in the file is answered
//! from it without a call.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{Answer, Ask, Question, Usage};

/// One answered question, as a line of the journal.
#[derive(Serialize, Deserialize)]
struct Entry {
    /// [`key`] of the question.
    key: String,
    text: String,
    calls: u64,
    input: u64,
    output: u64,
}

pub(super) struct Journaled<A> {
    inner: A,
    /// What the journal held when it was opened. A question is asked once
    /// per run, so answers written during this run are never looked up.
    answered: HashMap<String, Answer>,
    file: Mutex<File>,
}

impl<A> Journaled<A> {
    pub(super) fn open(inner: A, path: &Path) -> Result<Self> {
        let mut answered = HashMap::new();
        if path.exists() {
            let file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
            for (number, line) in BufReader::new(file).lines().enumerate() {
                let line = line.with_context(|| format!("reading {}", path.display()))?;
                // A run killed mid-write leaves its last line torn; that one
                // answer is asked again.
                let Ok(entry) = serde_json::from_str::<Entry>(&line) else {
                    eprintln!(
                        "{}:{}: not a journal entry; skipped",
                        path.display(),
                        number + 1
                    );
                    continue;
                };
                let usage = Usage {
                    calls: entry.calls,
                    input: entry.input,
                    output: entry.output,
                };
                answered.insert(
                    entry.key,
                    Answer {
                        text: entry.text,
                        usage,
                    },
                );
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self {
            inner,
            answered,
            file: Mutex::new(file),
        })
    }

    fn record(&self, key: String, answer: &Answer) -> Result<()> {
        let entry = Entry {
            key,
            text: answer.text.clone(),
            calls: answer.usage.calls,
            input: answer.usage.input,
            output: answer.usage.output,
        };
        let mut line = serde_json::to_string(&entry)?;
        line.push('\n');
        // One write per line, under the lock, so concurrent answers never
        // interleave.
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("journal lock poisoned"))?;
        file.write_all(line.as_bytes())?;
        file.flush()?;
        Ok(())
    }
}

impl<A: Ask> Ask for Journaled<A> {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        let key = key(question);
        if let Some(answer) = self.answered.get(&key) {
            return Ok(answer.clone());
        }
        let answer = self.inner.ask(question).await?;
        // A reply that breaks its schema's contract is not worth replaying:
        // the next run asks again.
        if question.schema.is_none()
            || serde_json::from_str::<serde_json::Value>(&answer.text).is_ok()
        {
            self.record(key, &answer)?;
        }
        Ok(answer)
    }
}

/// Everything that makes two questions the same question. `DefaultHasher`
/// is stable for a given toolchain; a new one only costs a re-ask.
fn key(question: &Question<'_>) -> String {
    let mut hasher = DefaultHasher::new();
    question.level.hash(&mut hasher);
    question.system.hash(&mut hasher);
    question.prompt.hash(&mut hasher);
    question.schema.map(ToString::to_string).hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
