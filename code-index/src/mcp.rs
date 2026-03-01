use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    pub result: Value,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub jsonrpc: String,
    pub id: Value,
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: i64,
    pub message: String,
}

pub fn ok(id: Value, result: Value) -> Value {
    serde_json::to_value(Response {
        jsonrpc: "2.0".to_string(),
        id,
        result,
    })
    .expect("Response serialization must not fail")
}

pub fn error(id: Value, code: i64, message: String) -> Value {
    serde_json::to_value(ErrorResponse {
        jsonrpc: "2.0".to_string(),
        id,
        error: ErrorBody { code, message },
    })
    .expect("ErrorResponse serialization must not fail")
}

pub fn tool_result(text: String) -> Value {
    serde_json::json!({
        "content": [{"type": "text", "text": text}]
    })
}

pub fn tool_error_result(text: String) -> Value {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "isError": true
    })
}
