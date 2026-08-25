//! Neutral data boundary between source decoding, caching, and photographic processing.

use image::DynamicImage;

use super::{AppliedPhotographicTransform, SourcePipeline};

pub(super) struct DecodedSource {
    pub(super) image: DynamicImage,
    pub(super) raw_linear: Option<DecodedRaw>,
    pub(super) source_dimensions: (u32, u32),
    pub(super) source_pipeline: SourcePipeline,
    pub(super) photographic: Option<AppliedPhotographicTransform>,
    pub(super) orientation_applied: Option<u8>,
}

#[derive(Clone)]
pub(super) struct DecodedRaw {
    pub(super) raw: rawler::RawImage,
    pub(super) orientation: Option<image::metadata::Orientation>,
}

impl DecodedRaw {
    pub(super) fn oriented_dimensions(&self) -> (u32, u32) {
        let width = self.raw.crop_area.map_or(self.raw.width, |area| area.d.w) as u32;
        let height = self.raw.crop_area.map_or(self.raw.height, |area| area.d.h) as u32;
        match self.orientation {
            Some(
                image::metadata::Orientation::Rotate90
                | image::metadata::Orientation::Rotate270
                | image::metadata::Orientation::Rotate90FlipH
                | image::metadata::Orientation::Rotate270FlipH,
            ) => (height, width),
            _ => (width, height),
        }
    }
}
