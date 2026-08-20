use std::{collections::HashMap, path::PathBuf, sync::Arc};

use gpui::RenderImage;
use image::{DynamicImage, Frame};

use crate::discovery::RequestId;
use crate::rendering::{RenderBackend, RenderMetadata, RenderRequest};

#[derive(Clone)]
pub enum PreviewState {
    Unloaded,
    Loading(RequestId),
    Loaded(Arc<DecodedPreview>),
    Error(Arc<str>),
}

pub struct DecodedPreview {
    pub image: Arc<RenderImage>,
    pub metadata: RenderMetadata,
}

/// Independent resource cache. Future thumbnails and metadata can follow the
/// same lifecycle without becoming part of the tree projection.
pub struct PreviewCache {
    entries: HashMap<PathBuf, PreviewState>,
    last_used: HashMap<PathBuf, u64>,
    capacity: usize,
    next_request: RequestId,
    access_clock: u64,
}

impl PreviewCache {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "preview cache capacity must be positive");
        Self {
            entries: HashMap::new(),
            last_used: HashMap::new(),
            capacity,
            next_request: 0,
            access_clock: 0,
        }
    }

    pub fn state(&self, path: &std::path::Path) -> PreviewState {
        self.entries
            .get(path)
            .cloned()
            .unwrap_or(PreviewState::Unloaded)
    }

    pub fn begin(&mut self, path: &std::path::Path) -> Option<RequestId> {
        if matches!(self.state(path), PreviewState::Loaded(_)) {
            self.touch(path);
            return None;
        }
        if !matches!(self.state(path), PreviewState::Unloaded) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        self.touch(path);
        self.entries
            .insert(path.to_path_buf(), PreviewState::Loading(request));
        Some(request)
    }

    pub fn finish(
        &mut self,
        path: &std::path::Path,
        request: RequestId,
        result: PreviewState,
    ) -> bool {
        if !matches!(self.state(path), PreviewState::Loading(active) if active == request) {
            return false;
        }
        let is_loaded = matches!(result, PreviewState::Loaded(_));
        self.entries.insert(path.to_path_buf(), result);
        if is_loaded {
            self.evict_over_capacity();
        } else {
            self.last_used.remove(path);
        }
        true
    }

    pub fn mark_used(&mut self, path: &std::path::Path) {
        if matches!(self.state(path), PreviewState::Loaded(_)) {
            self.touch(path);
        }
    }

    fn touch(&mut self, path: &std::path::Path) {
        self.access_clock += 1;
        self.last_used.insert(path.to_path_buf(), self.access_clock);
    }

    fn evict_over_capacity(&mut self) {
        while self
            .entries
            .values()
            .filter(|state| matches!(state, PreviewState::Loaded(_)))
            .count()
            > self.capacity
        {
            let evicted = self
                .entries
                .iter()
                .filter(|(_, state)| matches!(state, PreviewState::Loaded(_)))
                .min_by_key(|(id, _)| self.last_used.get(*id).copied().unwrap_or(0))
                .map(|(id, _)| id.clone());
            if let Some(evicted) = evicted {
                self.entries.remove(&evicted);
                self.last_used.remove(&evicted);
            }
        }
    }
}

/// Blocking image I/O and CPU-heavy decoding. Call only from a background executor.
pub fn decode_path(path: PathBuf, thumbnail_width: Option<u32>) -> PreviewState {
    let request = RenderRequest::new(path.clone(), thumbnail_width);
    match RenderBackend::default().render(&request) {
        Ok(rendered) => PreviewState::Loaded(Arc::new(DecodedPreview {
            image: Arc::new(render_image(rendered.image)),
            metadata: rendered.metadata,
        })),
        Err(error) => {
            PreviewState::Error(format!("Cannot decode {}: {error}", path.display()).into())
        }
    }
}

fn render_image(decoded: DynamicImage) -> RenderImage {
    let mut buffer = decoded.into_rgba8();
    for pixel in buffer.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    RenderImage::new(smallvec::smallvec![Frame::new(buffer)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use std::path::Path;

    fn image_state() -> PreviewState {
        let buffer = ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255]));
        PreviewState::Loaded(Arc::new(DecodedPreview {
            image: Arc::new(RenderImage::new(smallvec::smallvec![Frame::new(buffer)])),
            metadata: RenderMetadata {
                source_pipeline: crate::rendering::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (1, 1),
                output_dimensions: (1, 1),
                variant: crate::rendering::RenderVariant::Full,
                orientation_applied: None,
            },
        }))
    }

    fn load(cache: &mut PreviewCache, path: &std::path::Path) {
        let request = cache.begin(path).unwrap();
        assert!(cache.finish(path, request, image_state()));
    }

    #[test]
    fn evicts_the_least_recently_used_loaded_image() {
        let mut cache = PreviewCache::new(2);
        let first = Path::new("first");
        let second = Path::new("second");
        let third = Path::new("third");
        load(&mut cache, first);
        load(&mut cache, second);
        cache.mark_used(first);
        load(&mut cache, third);

        assert!(matches!(cache.state(first), PreviewState::Loaded(_)));
        assert!(matches!(cache.state(second), PreviewState::Unloaded));
        assert!(matches!(cache.state(third), PreviewState::Loaded(_)));
    }

    #[test]
    fn cheap_negative_results_do_not_consume_image_capacity() {
        let mut cache = PreviewCache::new(1);
        let missing = Path::new("missing");
        let image = Path::new("image");
        let request = cache.begin(missing).unwrap();
        cache.finish(missing, request, PreviewState::Error("missing".into()));
        load(&mut cache, image);

        assert!(matches!(cache.state(missing), PreviewState::Error(_)));
        assert!(matches!(cache.state(image), PreviewState::Loaded(_)));
    }

    #[test]
    fn completion_order_does_not_override_usage_order() {
        let mut cache = PreviewCache::new(1);
        let first = Path::new("first");
        let second = Path::new("second");
        let first_request = cache.begin(first).unwrap();
        let second_request = cache.begin(second).unwrap();
        cache.finish(second, second_request, image_state());
        cache.finish(first, first_request, image_state());

        assert!(matches!(cache.state(first), PreviewState::Unloaded));
        assert!(matches!(cache.state(second), PreviewState::Loaded(_)));
    }

    #[test]
    fn decodes_the_cr3_regression_sample() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/IMG_0052.CR3");
        if path.exists() {
            let result = decode_path(path, Some(64));
            if let PreviewState::Error(error) = result {
                panic!("CR3 sample did not decode: {error}");
            }
        }
    }
}
