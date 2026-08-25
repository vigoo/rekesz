use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ToneMapper {
    #[default]
    AcesFitted,
}

pub(super) fn map(tone_mapper: ToneMapper, value: f32) -> f32 {
    match tone_mapper {
        ToneMapper::AcesFitted => aces_fitted(value),
    }
}

fn aces_fitted(value: f32) -> f32 {
    if !value.is_finite() {
        return if value.is_sign_positive() { 1.0 } else { 0.0 };
    }
    ((value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests;
