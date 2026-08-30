use std::{
    path::PathBuf,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use image::{DynamicImage, RgbImage};
use test_r::test;

use super::*;
use crate::rendering::SourcePipeline;

fn key(name: &str) -> DecodedSourceKey {
    DecodedSourceKey {
        path: PathBuf::from(name),
        identity: SourceIdentity {
            length: name.len() as u64,
            modified_seconds: 1,
            modified_nanos: 2,
        },
    }
}

fn source(value: u8) -> Arc<DecodedSource> {
    Arc::new(DecodedSource {
        image: DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, image::Rgb([value; 3]))),
        raw_linear: None,
        source_dimensions: (1, 1),
        source_pipeline: SourcePipeline::Raw,
        photographic: None,
        orientation_applied: None,
        face_transform: crate::rendering::FaceTransform::IDENTITY,
    })
}

#[test]
fn concurrent_requests_coalesce_one_decode() {
    let cache = Arc::new(DecodedSourceCache::new(2));
    let decodes = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let worker = {
        let cache = cache.clone();
        let decodes = decodes.clone();
        let entered = entered.clone();
        let release = release.clone();
        std::thread::spawn(move || {
            cache.get_or_decode(key("raw"), &AtomicBool::new(false), || {
                decodes.fetch_add(1, Ordering::Relaxed);
                entered.wait();
                release.wait();
                Ok(source(7))
            })
        })
    };
    entered.wait();
    let contender = {
        let cache = cache.clone();
        let decodes = decodes.clone();
        std::thread::spawn(move || {
            cache.get_or_decode(key("raw"), &AtomicBool::new(false), || {
                decodes.fetch_add(1, Ordering::Relaxed);
                Ok(source(9))
            })
        })
    };
    std::thread::sleep(Duration::from_millis(20));
    release.wait();
    assert_eq!(
        worker
            .join()
            .unwrap()
            .unwrap()
            .image
            .to_rgb8()
            .get_pixel(0, 0)[0],
        7
    );
    assert_eq!(
        contender
            .join()
            .unwrap()
            .unwrap()
            .image
            .to_rgb8()
            .get_pixel(0, 0)[0],
        7
    );
    assert_eq!(decodes.load(Ordering::Relaxed), 1);
    assert_eq!(cache.hits(), 1);
}

#[test]
fn cancelled_waiter_exits_and_failed_decode_is_removed() {
    let cache = Arc::new(DecodedSourceCache::new(1));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = {
        let cache = cache.clone();
        std::thread::spawn(move || {
            cache.get_or_decode(key("raw"), &AtomicBool::new(false), || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Err(cancelled_image_error())
            })
        })
    };
    entered_rx.recv().unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(
        cache
            .get_or_decode(key("raw"), &cancelled, || Ok(source(3)))
            .is_err()
    );
    release_tx.send(()).unwrap();
    assert!(worker.join().unwrap().is_err());
    assert!(
        cache
            .get_or_decode(key("raw"), &AtomicBool::new(false), || Ok(source(3)))
            .is_ok()
    );
}

#[test]
fn hits_preserve_insertion_order_eviction_policy() {
    let cache = DecodedSourceCache::new(2);
    let cancelled = AtomicBool::new(false);
    for (name, value) in [("a", 1), ("b", 2)] {
        cache
            .get_or_decode(key(name), &cancelled, || Ok(source(value)))
            .unwrap();
    }
    cache
        .get_or_decode(key("a"), &cancelled, || panic!("a should hit"))
        .unwrap();
    cache
        .get_or_decode(key("c"), &cancelled, || Ok(source(3)))
        .unwrap();
    assert_eq!(cache.hits(), 1);
    cache
        .get_or_decode(key("b"), &cancelled, || panic!("b should remain cached"))
        .unwrap();
    let a_decodes = AtomicUsize::new(0);
    cache
        .get_or_decode(key("a"), &cancelled, || {
            a_decodes.fetch_add(1, Ordering::Relaxed);
            Ok(source(4))
        })
        .unwrap();
    assert_eq!(a_decodes.load(Ordering::Relaxed), 1);
    assert_eq!(cache.hits(), 2);
}
