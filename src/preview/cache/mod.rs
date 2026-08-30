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
    entries: HashMap<(PathBuf, RenderIdentity, RenderVariant), PreviewState>,
    last_used: HashMap<(PathBuf, RenderIdentity, RenderVariant), u64>,
    capacity: usize,
    next_request: RequestId,
    access_clock: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreviewCacheStats {
    pub capacity: usize,
    pub loaded: usize,
    pub loading: usize,
    pub errors: usize,
    pub estimated_memory_bytes: u64,
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
            .iter()
            .find(|((candidate, candidate_identity, _), _)| {
                candidate == path && *candidate_identity == identity
            })
            .map(|(_, state)| state)
            .cloned()
            .unwrap_or(PreviewState::Unloaded)
    }

    pub fn state_for_variant(
        &self,
        path: &std::path::Path,
        identity: RenderIdentity,
        variant: RenderVariant,
    ) -> PreviewState {
        self.entries
            .get(&(path.to_path_buf(), identity, variant))
            .cloned()
            .or_else(|| {
                self.entries
                    .iter()
                    .find_map(|((candidate, candidate_identity, _), state)| match state {
                        PreviewState::Loaded(preview)
                            if candidate == path
                                && *candidate_identity == identity
                                && variant_satisfies(preview.metadata.variant, variant) =>
                        {
                            Some(state.clone())
                        }
                        _ => None,
                    })
            })
            .unwrap_or(PreviewState::Unloaded)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.last_used.clear();
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        assert!(capacity > 0, "preview cache capacity must be positive");
        self.capacity = capacity;
        self.evict_over_capacity();
    }

    pub fn stats(&self) -> PreviewCacheStats {
        let mut stats = PreviewCacheStats {
            capacity: self.capacity,
            ..PreviewCacheStats::default()
        };
        for state in self.entries.values() {
            match state {
                PreviewState::Loaded(preview) => {
                    stats.loaded += 1;
                    let (width, height) = preview.metadata.output_dimensions;
                    stats.estimated_memory_bytes = stats.estimated_memory_bytes.saturating_add(
                        u64::from(width)
                            .saturating_mul(u64::from(height))
                            .saturating_mul(4),
                    );
                }
                PreviewState::Loading(_) => stats.loading += 1,
                PreviewState::Error(_) => stats.errors += 1,
                PreviewState::Unloaded => {}
            }
        }
        stats
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
        let satisfying = self.entries.iter().find_map(
            |((candidate, candidate_identity, cached_variant), state)| match state {
                PreviewState::Loaded(preview)
                    if candidate == path
                        && *candidate_identity == identity
                        && variant_satisfies(preview.metadata.variant, variant) =>
                {
                    Some(*cached_variant)
                }
                _ => None,
            },
        );
        if let Some(cached_variant) = satisfying {
            self.touch(path, identity, cached_variant);
            return None;
        }
        let key = (path.to_path_buf(), identity, variant);
        if matches!(
            self.state_for_variant(path, identity, variant),
            PreviewState::Loaded(_)
        ) {
            self.entries.remove(&key);
            self.last_used.remove(&key);
        }
        if !matches!(
            self.state_for_variant(path, identity, variant),
            PreviewState::Unloaded
        ) {
            return None;
        }
        self.next_request += 1;
        let request = self.next_request;
        self.touch(path, identity, variant);
        self.entries.insert(key, PreviewState::Loading(request));
        Some(request)
    }

    pub fn needs_work_for(
        &self,
        path: &std::path::Path,
        variant: RenderVariant,
        identity: RenderIdentity,
    ) -> bool {
        if self.entries.iter().any(
            |((candidate, candidate_identity, _), state)| {
                candidate == path
                    && *candidate_identity == identity
                    && matches!(state, PreviewState::Loaded(preview) if variant_satisfies(preview.metadata.variant, variant))
            },
        ) {
            return false;
        }
        match self.state_for_variant(path, identity, variant) {
            PreviewState::Loaded(preview) => !variant_satisfies(preview.metadata.variant, variant),
            PreviewState::Unloaded | PreviewState::Loading(_) => true,
            PreviewState::Error(_) => false,
        }
    }

    #[cfg(test)]
    pub fn cancel(&mut self, path: &std::path::Path, request: RequestId) -> bool {
        let identity = RenderIdentity::default();
        let Some(variant) = self.variant_for_request(path, identity, request) else {
            return false;
        };
        self.cancel_for(path, identity, variant, request)
    }

    pub fn cancel_for(
        &mut self,
        path: &std::path::Path,
        identity: RenderIdentity,
        variant: RenderVariant,
        request: RequestId,
    ) -> bool {
        if !matches!(self.state_for_variant(path, identity, variant), PreviewState::Loading(active) if active == request)
        {
            return false;
        }
        let key = (path.to_path_buf(), identity, variant);
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
        let identity = RenderIdentity::default();
        let Some(variant) = self.variant_for_request(path, identity, request) else {
            return false;
        };
        self.finish_for(path, identity, variant, request, result)
    }

    pub fn finish_for(
        &mut self,
        path: &std::path::Path,
        identity: RenderIdentity,
        variant: RenderVariant,
        request: RequestId,
        result: PreviewState,
    ) -> bool {
        if !matches!(self.state_for_variant(path, identity, variant), PreviewState::Loading(active) if active == request)
        {
            return false;
        }
        let key = (path.to_path_buf(), identity, variant);
        let is_terminal = matches!(result, PreviewState::Loaded(_) | PreviewState::Error(_));
        self.entries.insert(key.clone(), result);
        if is_terminal {
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
        let variants = self
            .entries
            .iter()
            .filter_map(|((candidate, candidate_identity, variant), state)| {
                (candidate == path
                    && *candidate_identity == identity
                    && matches!(state, PreviewState::Loaded(_)))
                .then_some(*variant)
            })
            .collect::<Vec<_>>();
        for variant in variants {
            self.touch(path, identity, variant);
        }
    }

    #[cfg(test)]
    fn variant_for_request(
        &self,
        path: &std::path::Path,
        identity: RenderIdentity,
        request: RequestId,
    ) -> Option<RenderVariant> {
        self.entries
            .iter()
            .find_map(|((candidate, candidate_identity, variant), state)| {
                (candidate == path
                    && *candidate_identity == identity
                    && matches!(state, PreviewState::Loading(active) if *active == request))
                .then_some(*variant)
            })
    }

    fn touch(&mut self, path: &std::path::Path, identity: RenderIdentity, variant: RenderVariant) {
        self.access_clock += 1;
        self.last_used
            .insert((path.to_path_buf(), identity, variant), self.access_clock);
    }

    fn evict_over_capacity(&mut self) {
        self.evict_state_over_capacity(|state| matches!(state, PreviewState::Loaded(_)));
        self.evict_state_over_capacity(|state| matches!(state, PreviewState::Error(_)));
    }

    fn evict_state_over_capacity(&mut self, matches_state: impl Fn(&PreviewState) -> bool) {
        while self
            .entries
            .values()
            .filter(|state| matches_state(state))
            .count()
            > self.capacity
        {
            let evicted = self
                .entries
                .iter()
                .filter(|(_, state)| matches_state(state))
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
