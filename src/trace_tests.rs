use super::secs;
use std::time::Duration;
use test_r::test;

#[test]
fn durations_format_for_trace_lines() {
    assert_eq!(secs(Duration::from_millis(400)), "0.4s");
    assert_eq!(secs(Duration::from_millis(12_340)), "12.3s");
    assert_eq!(secs(Duration::from_secs(242)), "4m2.0s");
}
