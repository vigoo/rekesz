use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecognitionPolicy {
    pub absolute_threshold: f32,
    pub nearest_exemplar_threshold: f32,
    pub second_best_margin: f32,
}

impl Default for RecognitionPolicy {
    fn default() -> Self {
        Self {
            absolute_threshold: 0.50,
            nearest_exemplar_threshold: 0.45,
            second_best_margin: 0.08,
        }
    }
}

pub(crate) fn accepted_margin(
    policy: RecognitionPolicy,
    centroid_score: f32,
    nearest_exemplar_score: f32,
    second_centroid_score: Option<f32>,
) -> Option<f32> {
    if centroid_score < policy.absolute_threshold
        || nearest_exemplar_score < policy.nearest_exemplar_threshold
    {
        return None;
    }
    let margin = second_centroid_score.map_or(2.0, |second| centroid_score - second);
    (margin >= policy.second_best_margin).then_some(margin)
}

pub(crate) fn normalize(values: &mut [f32]) -> Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        bail!("embedding has zero or non-finite norm");
    }
    values.iter_mut().for_each(|value| *value /= norm);
    Ok(())
}

pub(crate) fn robust_centroid(exemplars: &[Vec<f32>]) -> Result<Vec<f32>> {
    let dimensions = exemplars
        .first()
        .map(Vec::len)
        .context("cannot build a prototype without exemplars")?;
    let mut initial = vec![0.0_f32; dimensions];
    for exemplar in exemplars {
        if exemplar.len() != dimensions {
            bail!("confirmed embeddings have incompatible dimensions");
        }
        for (sum, value) in initial.iter_mut().zip(exemplar) {
            *sum += value;
        }
    }
    normalize(&mut initial).context("confirmed embeddings have no stable centroid")?;
    if exemplars.len() < 4 {
        return Ok(initial);
    }
    let mut ranked = exemplars
        .iter()
        .map(|value| (dot(value, &initial), value))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
    let retained = ranked.len() - ranked.len() / 4;
    let mut robust = vec![0.0_f32; dimensions];
    for (_, exemplar) in ranked.into_iter().take(retained) {
        for (sum, value) in robust.iter_mut().zip(exemplar) {
            *sum += value;
        }
    }
    normalize(&mut robust).context("confirmed embeddings have no robust centroid")?;
    Ok(robust)
}

pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
#[path = "recognition_policy_tests.rs"]
mod tests;
