//! `agent::classify` reads rig's provider errors as rate signals.

use std::time::Duration;

use desloppify::agent::classify;
use desloppify::rate_limit::{BoxError, Signal};
use rig_core::ProviderError;

fn reply(status: u16, retry_after: Option<&str>) -> BoxError {
    let status = http::StatusCode::from_u16(status).unwrap();
    let headers = retry_after.map(|value| {
        let mut headers = http::HeaderMap::new();
        headers.insert("retry-after", value.parse().unwrap());
        headers
    });
    Box::new(ProviderError::from_http_response(status, "").with_response_headers(headers))
}

#[test]
fn a_429_is_a_throttle_carrying_its_retry_after_seconds() {
    assert_eq!(
        classify(&*reply(429, Some("7"))),
        Signal::Throttled {
            retry_after: Some(Duration::from_secs(7))
        }
    );
}

#[test]
fn a_retry_after_date_is_ignored() {
    assert_eq!(
        classify(&*reply(429, Some("Wed, 21 Oct 2015 07:28:00 GMT"))),
        Signal::Throttled { retry_after: None }
    );
}

#[test]
fn an_outage_is_a_failure_not_a_throttle() {
    assert_eq!(
        classify(&*reply(503, Some("3"))),
        Signal::Failed {
            retry_after: Some(Duration::from_secs(3))
        }
    );
}

#[test]
fn an_error_that_is_not_the_providers_is_a_failure() {
    let other: BoxError = Box::new(std::io::Error::other("reset"));
    assert_eq!(classify(&*other), Signal::Failed { retry_after: None });
}

mod journal {
    use std::sync::atomic::{AtomicU64, Ordering};

    use anyhow::Result;
    use desloppify::agent::{Answer, Ask, Question, Usage, journaled};
    use desloppify::model::ModelLevel;

    /// Answers `text`, counting the calls that reached it.
    struct Counting {
        text: &'static str,
        calls: &'static AtomicU64,
    }

    impl Ask for Counting {
        async fn ask(&self, _: &Question<'_>) -> Result<Answer> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Answer {
                text: self.text.to_owned(),
                usage: Usage {
                    calls: 1,
                    input: 10,
                    output: 2,
                },
            })
        }
    }

    fn question<'a>(prompt: &'a str, schema: Option<&'a serde_json::Value>) -> Question<'a> {
        Question {
            level: ModelLevel::try_from(1).unwrap(),
            system: "system",
            prompt,
            schema,
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("desloppify-journal-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("journal.jsonl");
        let _removed = std::fs::remove_file(&path);
        path
    }

    #[tokio::test]
    async fn a_rerun_answers_from_the_journal_without_a_call() {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let path = scratch("rerun");
        let schema = serde_json::json!({"type": "object"});
        let asked = |text| Counting {
            text,
            calls: &CALLS,
        };

        let first = journaled(asked("{\"a\":1}"), &path).unwrap();
        let answer = first.ask(&question("p", Some(&schema))).await.unwrap();
        drop(first);
        let second = journaled(asked("{\"a\":2}"), &path).unwrap();
        let replayed = second.ask(&question("p", Some(&schema))).await.unwrap();

        assert_eq!(replayed, answer);
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        second.ask(&question("q", Some(&schema))).await.unwrap();
        assert_eq!(CALLS.load(Ordering::SeqCst), 2, "a new question is asked");
    }

    #[tokio::test]
    async fn a_reply_that_breaks_its_schema_is_asked_again() {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let path = scratch("broken");
        let schema = serde_json::json!({"type": "object"});
        for _ in 0..2 {
            let ask = journaled(
                Counting {
                    text: "I need to see the code",
                    calls: &CALLS,
                },
                &path,
            )
            .unwrap();
            ask.ask(&question("p", Some(&schema))).await.unwrap();
        }
        assert_eq!(CALLS.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_torn_last_line_is_skipped_and_the_rest_replayed() {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let path = scratch("torn");
        let ask = journaled(
            Counting {
                text: "free text",
                calls: &CALLS,
            },
            &path,
        )
        .unwrap();
        ask.ask(&question("p", None)).await.unwrap();
        drop(ask);
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut file, b"{\"key\":\"trunc").unwrap();
        let ask = journaled(
            Counting {
                text: "free text",
                calls: &CALLS,
            },
            &path,
        )
        .unwrap();
        ask.ask(&question("p", None)).await.unwrap();
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    }
}
