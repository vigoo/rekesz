use std::sync::atomic::AtomicBool;

use image::RgbaImage;

use super::warnings;

mod clarity;
mod crop;
mod detail;
mod global;
mod grain;
mod heal;
mod homography;
mod lens_distortion;
mod log_detail;
mod monochrome;
mod parametric_curve;
mod presence;
mod vignette;

pub(super) use clarity::Clarity;
pub(super) use crop::CropAction;
pub(super) use detail::Detail;
pub(super) use global::Global;
pub(super) use grain::Grain;
pub(super) use heal::Heal;
pub(super) use homography::Homography;
pub(super) use lens_distortion::LensDistortion;
pub(super) use monochrome::Monochrome;
pub(super) use parametric_curve::ParametricCurve;
pub(super) use presence::Presence;
pub(super) use vignette::Vignette;

/// A declarative develop operation. Implementations contain both the action's
/// parameters and its isolated pixel transformation.
pub(super) trait DevelopAction {
    fn action_mask(&self) -> u64;

    fn fidelity_mask(&self) -> u64 {
        warnings::APPROXIMATED
    }

    fn apply(&self, image: &mut RgbaImage, cancelled: &AtomicBool) -> image::ImageResult<()>;
}
