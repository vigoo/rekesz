use std::{collections::HashMap, path::PathBuf, sync::Arc};

use gpui::RenderImage;
use image::{DynamicImage, Frame, ImageDecoder};

use crate::discovery::{NodeId, RequestId};

#[derive(Clone)]
pub enum PreviewState {
    Unloaded,
    Loading(RequestId),
    Loaded(Arc<DecodedPreview>),
    Missing,
    Error(Arc<str>),
}

pub struct DecodedPreview {
    pub image: Arc<RenderImage>,
    pub file_name: Arc<str>,
}

/// Independent resource cache. Future thumbnails and metadata can follow the
/// same lifecycle without becoming part of the tree projection.
pub struct PreviewCache {
    entries: HashMap<NodeId, PreviewState>,
    last_used: HashMap<NodeId, u64>,
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

    pub fn state(&self, id: NodeId) -> PreviewState {
        self.entries
            .get(&id)
            .cloned()
            .unwrap_or(PreviewState::Unloaded)
    }

    pub fn begin(&mut self, id: NodeId) -> Option<RequestId> {
        if matches!(self.state(id), PreviewState::Loaded(_)) {
            self.touch(id);
            return None;
        }
        if !matches!(self.state(id), PreviewState::Unloaded) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        self.touch(id);
        self.entries.insert(id, PreviewState::Loading(request));
        Some(request)
    }

    pub fn finish(&mut self, id: NodeId, request: RequestId, result: PreviewState) -> bool {
        if !matches!(self.state(id), PreviewState::Loading(active) if active == request) {
            return false;
        }
        let is_loaded = matches!(result, PreviewState::Loaded(_));
        self.entries.insert(id, result);
        if is_loaded {
            self.evict_over_capacity();
        } else {
            self.last_used.remove(&id);
        }
        true
    }

    pub fn mark_used(&mut self, id: NodeId) {
        if matches!(self.state(id), PreviewState::Loaded(_)) {
            self.touch(id);
        }
    }

    fn touch(&mut self, id: NodeId) {
        self.access_clock += 1;
        self.last_used.insert(id, self.access_clock);
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
                .min_by_key(|(id, _)| self.last_used.get(id).copied().unwrap_or(0))
                .map(|(id, _)| *id);
            if let Some(evicted) = evicted {
                self.entries.remove(&evicted);
                self.last_used.remove(&evicted);
            }
        }
    }
}

/// Blocking image I/O and CPU-heavy decoding. Call only from a background executor.
pub fn decode_first(paths: Vec<PathBuf>) -> PreviewState {
    if paths.is_empty() {
        return PreviewState::Missing;
    }
    let mut last_error = None;
    for path in paths {
        match oriented_render_image(&path) {
            Ok(image) => {
                let file_name = path
                    .file_name()
                    .unwrap_or(path.as_os_str())
                    .to_string_lossy()
                    .into_owned()
                    .into();
                return PreviewState::Loaded(Arc::new(DecodedPreview {
                    image: Arc::new(image),
                    file_name,
                }));
            }
            Err(error) => last_error = Some(format!("Cannot decode {}: {error}", path.display())),
        }
    }
    PreviewState::Error(
        last_error
            .unwrap_or_else(|| "No decodable image found".into())
            .into(),
    )
}

fn oriented_render_image(path: &std::path::Path) -> image::ImageResult<RenderImage> {
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut decoded = DynamicImage::from_decoder(decoder)?;
    decoded.apply_orientation(orientation);
    let mut buffer = decoded.into_rgba8();
    for pixel in buffer.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    Ok(RenderImage::new(smallvec::smallvec![Frame::new(buffer)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn image_state() -> PreviewState {
        let buffer = ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255]));
        PreviewState::Loaded(Arc::new(DecodedPreview {
            image: Arc::new(RenderImage::new(smallvec::smallvec![Frame::new(buffer)])),
            file_name: "test.png".into(),
        }))
    }

    fn load(cache: &mut PreviewCache, id: NodeId) {
        let request = cache.begin(id).unwrap();
        assert!(cache.finish(id, request, image_state()));
    }

    #[test]
    fn evicts_the_least_recently_used_loaded_image() {
        let mut cache = PreviewCache::new(2);
        let first = NodeId(1);
        let second = NodeId(2);
        let third = NodeId(3);
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
        let missing = NodeId(1);
        let image = NodeId(2);
        let request = cache.begin(missing).unwrap();
        cache.finish(missing, request, PreviewState::Missing);
        load(&mut cache, image);

        assert!(matches!(cache.state(missing), PreviewState::Missing));
        assert!(matches!(cache.state(image), PreviewState::Loaded(_)));
    }

    #[test]
    fn completion_order_does_not_override_usage_order() {
        let mut cache = PreviewCache::new(1);
        let first = NodeId(1);
        let second = NodeId(2);
        let first_request = cache.begin(first).unwrap();
        let second_request = cache.begin(second).unwrap();
        cache.finish(second, second_request, image_state());
        cache.finish(first, first_request, image_state());

        assert!(matches!(cache.state(first), PreviewState::Unloaded));
        assert!(matches!(cache.state(second), PreviewState::Loaded(_)));
    }
}
