use std::{collections::BTreeMap, io};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut artifacts = BTreeMap::new();
    for line in io::stdin().lines() {
        let line = line?;
        let value: serde_json::Value = serde_json::from_str(&line)?;
        if value.get("reason").and_then(|value| value.as_str()) != Some("compiler-artifact")
            || value
                .pointer("/profile/test")
                .and_then(|value| value.as_bool())
                != Some(true)
        {
            continue;
        }
        let Some(executable) = value.get("executable").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(name) = value
            .pointer("/target/name")
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        if artifacts
            .insert(name.to_owned(), executable.to_owned())
            .is_some()
        {
            return Err(format!("duplicate test target in Cargo artifacts: {name}").into());
        }
    }
    for (name, executable) in artifacts {
        println!("{name}={executable}");
    }
    Ok(())
}
