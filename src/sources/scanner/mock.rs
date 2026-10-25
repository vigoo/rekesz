//! Shared mock eSCL server for scanner tests: a small but exact
//! HTTP/1.1 responder (correct Content-Length or chunked framing — reqwest
//! rejects sloppy replies) with recorded requests, usable from both the
//! scanner module tests and the import-view tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Mock eSCL server
// ---------------------------------------------------------------------------

pub(crate) struct MockReply {
    pub(crate) status: u16,
    pub(crate) content_type: Option<String>,
    pub(crate) location: Option<String>,
    pub(crate) chunked: bool,
    /// Declare a larger Content-Length than the body written, then close:
    /// models a connection drop mid-transfer.
    pub(crate) truncated: bool,
    pub(crate) body: Vec<u8>,
}

impl MockReply {
    pub(crate) fn status(status: u16) -> Self {
        Self {
            status,
            content_type: None,
            location: None,
            chunked: false,
            truncated: false,
            body: Vec::new(),
        }
    }

    pub(crate) fn created(location: &str) -> Self {
        Self {
            status: 201,
            location: Some(location.to_string()),
            ..Self::status(201)
        }
    }

    pub(crate) fn image(content_type: &str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type: Some(content_type.to_string()),
            body,
            ..Self::status(200)
        }
    }

    pub(crate) fn xml(body: &str) -> Self {
        Self::image("text/xml", body.as_bytes().to_vec())
    }

    pub(crate) fn chunked(mut self) -> Self {
        self.chunked = true;
        self
    }

    pub(crate) fn truncated(mut self) -> Self {
        self.truncated = true;
        self
    }
}

pub(crate) type MockHandler = Box<dyn FnMut(&str, &str, &str, &[u8]) -> MockReply + Send>;

/// Recorded requests of a mock server run: (method, path, Host header).
pub(crate) type RecordedRequests = Arc<Mutex<Vec<(String, String, String)>>>;

/// Spawn a single-threaded HTTP/1.1 responder on 127.0.0.1:0. One request
/// per connection, `Connection: close` framing, correct Content-Length or
/// chunked encoding — reqwest rejects sloppy replies, so the framing here
/// is exact. Returns the eSCL base URL and the recorded requests.
pub(crate) fn spawn_mock(handler: MockHandler) -> (String, RecordedRequests) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let recorded: RecordedRequests = Arc::new(Mutex::new(Vec::new()));
    let recorded_clone = Arc::clone(&recorded);
    let handler = Arc::new(Mutex::new(handler));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            let head_end = loop {
                match stream.read(&mut chunk) {
                    Ok(0) => break buffer.len(),
                    Ok(read) => {
                        buffer.extend_from_slice(&chunk[..read]);
                        if let Some(position) =
                            buffer.windows(4).position(|window| window == b"\r\n\r\n")
                        {
                            break position + 4;
                        }
                    }
                    Err(_) => break buffer.len(),
                }
            };
            let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
            let mut lines = head.split("\r\n");
            let request_line = lines.next().unwrap_or("");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let path = parts.next().unwrap_or("").to_string();
            let mut content_length = 0_usize;
            let mut host = String::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if let Some(value) = lower.strip_prefix("content-length:") {
                    content_length = value.trim().parse().unwrap_or(0);
                }
                if let Some(value) = lower.strip_prefix("host:") {
                    host = value.trim().to_string();
                }
            }
            let mut body = buffer[head_end..].to_vec();
            while body.len() < content_length {
                match stream.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => body.extend_from_slice(&chunk[..read]),
                    Err(_) => break,
                }
            }
            recorded_clone
                .lock()
                .unwrap()
                .push((method.clone(), path.clone(), host.clone()));
            let reply = (handler.lock().unwrap())(&method, &path, &host, &body);
            let status_text = match reply.status {
                200 => "OK",
                201 => "Created",
                404 => "Not Found",
                409 => "Conflict",
                410 => "Gone",
                500 => "Internal Server Error",
                503 => "Service Unavailable",
                _ => "Status",
            };
            let mut head_out = format!(
                "HTTP/1.1 {} {}\r\nConnection: close\r\n",
                reply.status, status_text
            );
            if let Some(content_type) = &reply.content_type {
                head_out.push_str(&format!("Content-Type: {content_type}\r\n"));
            }
            if let Some(location) = &reply.location {
                head_out.push_str(&format!("Location: {location}\r\n"));
            }
            if reply.chunked {
                head_out.push_str("Transfer-Encoding: chunked\r\n\r\n");
                let _ = stream.write_all(head_out.as_bytes());
                for frame in reply.body.chunks(16 * 1024) {
                    let _ = stream.write_all(format!("{:x}\r\n", frame.len()).as_bytes());
                    let _ = stream.write_all(frame);
                    let _ = stream.write_all(b"\r\n");
                }
                let _ = stream.write_all(b"0\r\n\r\n");
            } else if reply.truncated {
                // Header promises more than is written; the stream then
                // closes, so the client sees a short read.
                head_out.push_str("Content-Length: 100000\r\n\r\n");
                let _ = stream.write_all(head_out.as_bytes());
                let _ = stream.write_all(&reply.body);
            } else {
                head_out.push_str(&format!("Content-Length: {}\r\n\r\n", reply.body.len()));
                let _ = stream.write_all(head_out.as_bytes());
                let _ = stream.write_all(&reply.body);
            }
        }
    });
    (format!("http://{address}/eSCL"), recorded)
}
