use std::{collections::HashMap, path::PathBuf, sync::Arc};

use gpui::RenderImage;

use crate::{
    discovery::RequestId,
    rendering::{RenderIdentity, RenderMetadata, RenderVariant},
};

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

/// UI-owned cache of GPUI images and in-flight request state.
pub struct PreviewCache {
    entries: HashMap<(PathBuf, RenderIdentity), PreviewState>,
    last_used: HashMap<(PathBuf, RenderIdentity), u64>,
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

    #[cfg(test)]
    pub fn state(&self, path: &std::path::Path) -> PreviewState {
        self.state_for(path, RenderIdentity::default())
    }

    pub fn state_for(&self, path: &std::path::Path, identity: RenderIdentity) -> PreviewState {
        self.entries
            .get(&(path.to_path_buf(), identity))
            .cloned()
            .unwrap_or(PreviewState::Unloaded)
    }

    #[cfg(test)]
    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_used.clear();
    }

    #[cfg(test)]
    pub fn begin(&mut self, path: &std::path::Path, variant: RenderVariant) -> Option<RequestId> {
        self.begin_for(path, variant, RenderIdentity::default())
    }

    pub fn begin_for(
        &mut self,
        path: &std::path::Path,
        variant: RenderVariant,
        identity: RenderIdentity,
    ) -> Option<RequestId> {
        if matches!(self.state_for(path, identity), PreviewState::Loaded(ref preview) if variant_satisfies(preview.metadata.variant, variant))
        {
            self.touch(path, identity);
            return None;
        }
        let key = (path.to_path_buf(), identity);
        if matches!(self.state_for(path, identity), PreviewState::Loaded(_)) {
            self.entries.remove(&key);
            self.last_used.remove(&key);
        }
        if !matches!(self.state_for(path, identity), PreviewState::Unloaded) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        self.touch(path, identity);
        self.entries.insert(key, PreviewState::Loading(request));
        Some(request)
    }

    pub fn needs_work_for(
        &self,
        path: &std::path::Path,
        variant: RenderVariant,
        identity: RenderIdentity,
    ) -> bool {
        match self.state_for(path, identity) {
            PreviewState::Loaded(preview) => !variant_satisfies(preview.metadata.variant, variant),
            PreviewState::Unloaded | PreviewState::Loading(_) => true,
            PreviewState::Error(_) => false,
        }
    }

    #[cfg(test)]
    pub fn cancel(&mut self, path: &std::path::Path, request: RequestId) -> bool {
        self.cancel_for(path, RenderIdentity::default(), request)
    }

    pub fn cancel_for(
        &mut self,
        path: &std::path::Path,
        identity: RenderIdentity,
        request: RequestId,
    ) -> bool {
        if !matches!(self.state_for(path, identity), PreviewState::Loading(active) if active == request)
        {
            return false;
        }
        let key = (path.to_path_buf(), identity);
        self.entries.remove(&key);
        self.last_used.remove(&key);
        true
    }

    #[cfg(test)]
    pub fn finish(
        &mut self,
        path: &std::path::Path,
        request: RequestId,
        result: PreviewState,
    ) -> bool {
        self.finish_for(path, RenderIdentity::default(), request, result)
    }

    pub fn finish_for(
        &mut self,
        path: &std::path::Path,
        identity: RenderIdentity,
        request: RequestId,
        result: PreviewState,
    ) -> bool {
        if !matches!(self.state_for(path, identity), PreviewState::Loading(active) if active == request)
        {
            return false;
        }
        let key = (path.to_path_buf(), identity);
        let is_loaded = matches!(result, PreviewState::Loaded(_));
        self.entries.insert(key.clone(), result);
        if is_loaded {
            self.evict_over_capacity();
        } else {
            self.last_used.remove(&key);
        }
        true
    }

    #[cfg(test)]
    pub fn mark_used(&mut self, path: &std::path::Path) {
        self.mark_used_for(path, RenderIdentity::default());
    }

    pub fn mark_used_for(&mut self, path: &std::path::Path, identity: RenderIdentity) {
        if matches!(self.state_for(path, identity), PreviewState::Loaded(_)) {
            self.touch(path, identity);
        }
    }

    fn touch(&mut self, path: &std::path::Path, identity: RenderIdentity) {
        self.access_clock += 1;
        self.last_used
            .insert((path.to_path_buf(), identity), self.access_clock);
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

fn variant_satisfies(cached: RenderVariant, requested: RenderVariant) -> bool {
    match (cached, requested) {
        (
            RenderVariant::Preview {
                max_width: cached_width,
                max_height: cached_height,
            },
            RenderVariant::Preview {
                max_width,
                max_height,
            },
        ) => cached_width >= max_width && cached_height >= max_height,
        _ => cached == requested,
    }
}

#[cfg(test)]
mod tests;
