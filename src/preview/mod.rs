mod cache;

use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

use gpui::RenderImage;
use image::{DynamicImage, Frame};

use crate::rendering::pipeline::RenderPipeline as CoreRenderPipeline;
#[cfg(test)]
use crate::rendering::{RenderIdentity, RenderVariant};

use crate::rendering::pipeline::RenderInvocation;
pub use cache::{DecodedPreview, PreviewCache, PreviewCacheStats, PreviewState};

/// UI adapter around the renderer. Core execution returns a renderer-owned
/// image; this layer performs the GPUI conversion and exposes preview state.
#[derive(Clone)]
pub struct PreviewRenderer {
    pub(crate) pipeline: CoreRenderPipeline,
}

impl PreviewRenderer {
    pub fn with_source_cache_capacity(
        cache_root: PathBuf,
        parallelism: usize,
        raw_parallelism: usize,
        source_cache_capacity: usize,
    ) -> Self {
        Self {
            pipeline: CoreRenderPipeline::with_source_cache_capacity(
                cache_root,
                parallelism,
                raw_parallelism,
                source_cache_capacity,
            ),
        }
    }

    pub fn stats(&self) -> crate::rendering::pipeline::PipelineStats {
        self.pipeline.stats()
    }

    pub fn disk_cache_usage(&self) -> Result<crate::rendering::pipeline::DiskCacheUsage, String> {
        self.pipeline.disk_cache_usage()
    }

    pub fn clear_disk_cache(&self) -> Result<(), String> {
        self.pipeline.clear_disk_cache()
    }

    pub fn set_source_cache_capacity(&self, capacity: usize) {
        self.pipeline.set_source_cache_capacity(capacity);
    }

    #[cfg(test)]
    pub fn render(
        &self,
        path: PathBuf,
        variant: RenderVariant,
        interactive: bool,
        durable_cache_write: bool,
        cancelled: &Arc<AtomicBool>,
    ) -> PreviewState {
        self.render_with_recipe(
            RenderInvocation {
                path,
                variant,
                render_identity: RenderIdentity::default(),
                recipe: None,
                interactive,
                durable_cache_write,
            },
            cancelled,
        )
    }

    pub fn render_with_recipe(
        &self,
        invocation: RenderInvocation<'_>,
        cancelled: &Arc<AtomicBool>,
    ) -> PreviewState {
        match self.pipeline.render_with_recipe(invocation, cancelled) {
            Ok(rendered) => {
                let image = Arc::new(render_image(&rendered.image));
                PreviewState::Loaded(Arc::new(DecodedPreview {
                    image,
                    metadata: rendered.metadata.clone(),
                }))
            }
            Err(error) => PreviewState::Error(error),
        }
    }
}

pub(crate) fn render_image(decoded: &DynamicImage) -> RenderImage {
    let mut buffer = decoded.to_rgba8();
    for pixel in buffer.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    RenderImage::new(smallvec::smallvec![Frame::new(buffer)])
}

#[cfg(test)]
mod tests;
