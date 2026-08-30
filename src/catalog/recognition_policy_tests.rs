use super::*;
use test_r::test;

#[test]
fn four_exemplars_trim_the_lowest_similarity_quarter() {
    let exemplars = vec![
        vec![1.0, 0.0],
        vec![0.99, 0.01],
        vec![0.98, 0.02],
        vec![-1.0, 0.0],
    ];
    let centroid = robust_centroid(&exemplars).unwrap();
    assert!(centroid[0] > 0.999);
    assert!(centroid[1] > 0.0);
}

#[test]
fn acceptance_uses_inclusive_absolute_exemplar_and_margin_boundaries() {
    let policy = RecognitionPolicy {
        absolute_threshold: 0.75,
        nearest_exemplar_threshold: 0.75,
        second_best_margin: 0.25,
    };
    assert_eq!(accepted_margin(policy, 0.75, 0.75, Some(0.50)), Some(0.25));
    assert_eq!(accepted_margin(policy, 0.749, 0.90, Some(0.0)), None);
    assert_eq!(accepted_margin(policy, 0.90, 0.749, Some(0.0)), None);
    assert_eq!(accepted_margin(policy, 0.75, 0.75, Some(0.501)), None);
}
