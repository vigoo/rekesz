//! Epson NetScanMonitor push scan: scanner-panel-initiated scans received
//! as announcements (UDP 2968), answered with keepalives, accepted as a
//! SOAP push trigger, and executed as an ESC/I-2 session. See
//! scanner-push-scan-plan.md; protocol facts are byte-exact ports from the
//! MIT-licensed epson2paperless project.

pub mod announce;
pub mod engine;
pub mod trigger;
pub mod wire;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
