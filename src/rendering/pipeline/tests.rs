use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use image::{DynamicImage, ImageBuffer, Rgba};
use test_r::test;

use super::*;
use crate::rendering::{RenderMetadata, SourcePipeline};

#[test]
fn interactive_permits_bypass_background_reserve() {
    let pool = Arc::new(PermitPool::new(4, 2));
    let cancellation = AtomicBool::new(false);
    let background = pool.acquire(2, false, &cancellation).unwrap();
    let (sender, receiver) = mpsc::channel();
    let contender = {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let cancellation = AtomicBool::new(false);
            let _permit = pool.acquire(2, false, &cancellation).unwrap();
            sender.send("background").unwrap();
        })
    };
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    let interactive = pool.acquire(2, true, &cancellation).unwrap();
    drop(interactive);
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    drop(background);
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
        "background"
    );
    contender.join().unwrap();
}

#[test]
fn permit_wait_is_cooperatively_cancellable() {
    let pool = Arc::new(PermitPool::new(1, 0));
    let cancellation = Arc::new(AtomicBool::new(false));
    let held = pool.acquire(1, false, &cancellation).unwrap();
    let waiter = {
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        std::thread::spawn(move || pool.acquire(1, false, &cancellation).is_none())
    };
    std::thread::sleep(Duration::from_millis(30));
    cancellation.store(true, Ordering::Release);
    assert!(waiter.join().unwrap());
    drop(held);
}

fn cache_write_fixture() -> (tempfile::TempDir, RenderBackend, CacheWrite) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.png");
    std::fs::write(&source, b"identity").unwrap();
    let backend = RenderBackend::new(directory.path().join("cache"));
    let request = RenderRequest::new(source, RenderVariant::Thumbnail { width: 4 });
    let identity = backend.source_identity(&request).unwrap().unwrap();
    let rendered = Arc::new(RenderedImage {
        image: DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([1, 2, 3, 255]))),
        metadata: RenderMetadata {
            source_pipeline: SourcePipeline::Standard,
            photographic: None,
            source_dimensions: (1, 1),
            output_dimensions: (1, 1),
            variant: request.variant,
            orientation_applied: None,
            develop: None,
        },
    });
    (
        directory,
        backend,
        CacheWrite {
            request,
            identity,
            rendered,
        },
    )
}

#[test]
fn durable_background_writes_wait_for_bounded_queue_capacity() {
    let (_directory, _backend, write) = cache_write_fixture();
    let make_write = || CacheWrite {
        request: write.request.clone(),
        identity: write.identity.clone(),
        rendered: write.rendered.clone(),
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    assert!(sender.try_send(make_write()).is_ok());
    let writer = Arc::new(CacheWriter {
        sender: Mutex::new(Some(sender)),
        worker: Mutex::new(None),
        metrics: Arc::new(PipelineMetrics::default()),
    });
    let cancellation = Arc::new(AtomicBool::new(false));
    let waiter = {
        let writer = writer.clone();
        let cancellation = cancellation.clone();
        let write = make_write();
        std::thread::spawn(move || writer.write(write, true, &cancellation))
    };
    std::thread::sleep(Duration::from_millis(30));
    assert!(!waiter.is_finished());
    receiver.recv().unwrap();
    waiter.join().unwrap();
    receiver.recv_timeout(Duration::from_secs(1)).unwrap();
}

#[test]
fn writer_drop_drains_queue_and_cache_reopens() {
    let (directory, backend, write) = cache_write_fixture();
    let request = write.request.clone();
    let identity = write.identity.clone();
    let metrics = Arc::new(PipelineMetrics::default());
    let (sender, receiver) = mpsc::sync_channel::<CacheWrite>(2);
    let worker_backend = backend.clone();
    let worker = std::thread::spawn(move || {
        while let Ok(write) = receiver.recv() {
            worker_backend
                .store_cache(&write.request, &write.identity, &write.rendered)
                .unwrap();
        }
    });
    let writer = CacheWriter {
        sender: Mutex::new(Some(sender)),
        worker: Mutex::new(Some(worker)),
        metrics,
    };
    writer.write(write, true, &AtomicBool::new(false));
    drop(writer);

    let reopened = RenderBackend::new(directory.path().join("cache"));
    assert!(
        reopened
            .lookup_cache(&request, Some(&identity))
            .unwrap()
            .is_some()
    );
}

#[test]
fn staged_pipeline_persists_generated_thumbnail() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.png");
    DynamicImage::ImageRgba8(ImageBuffer::from_pixel(8, 8, Rgba([1, 2, 3, 255])))
        .save(&source)
        .unwrap();
    let pipeline =
        RenderPipeline::with_source_cache_capacity(directory.path().join("cache"), 2, 2, 1);
    let variant = RenderVariant::Thumbnail { width: 4 };
    assert!(
        pipeline
            .render(
                source.clone(),
                variant,
                true,
                false,
                &Arc::new(AtomicBool::new(false))
            )
            .is_ok()
    );
    assert_eq!(pipeline.stats().cache_misses, 1);
    let request = RenderRequest::new(source, variant);
    let identity = pipeline.backend.source_identity(&request).unwrap().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if pipeline
            .backend
            .lookup_cache(&request, Some(&identity))
            .unwrap()
            .is_some()
        {
            return;
        }
        std::thread::yield_now();
    }
    panic!("deferred cache write did not become visible");
}
