use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

use super::decoding_error;
use crate::rendering::SourcePipeline;
use crate::rendering::decoded::DecodedSource;

pub(super) fn decode(
    path: &Path,
    cancelled: &Arc<AtomicBool>,
    analysis_memory_budget: Option<usize>,
) -> image::ImageResult<DecodedSource> {
    decode_image(path, cancelled, analysis_memory_budget)
        .map(|(image, orientation, face_transform)| DecodedSource {
            source_dimensions: (image.width(), image.height()),
            image,
            raw_linear: None,
            source_pipeline: SourcePipeline::Heif,
            photographic: None,
            orientation_applied: Some(orientation),
            face_transform,
        })
        .map_err(decoding_error)
}

fn decode_image(
    path: &Path,
    cancelled: &Arc<AtomicBool>,
    analysis_memory_budget: Option<usize>,
) -> Result<(DynamicImage, u8, crate::rendering::FaceTransform), String> {
    let compressed_bytes = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if !heif_pre_read_admitted(compressed_bytes, analysis_memory_budget) {
        return Err("HEIF source exceeds the pre-read analysis memory budget".into());
    }
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if cancelled.load(Ordering::Acquire) {
        return Err("render cancelled".into());
    }
    let context = HeifContext::read_from_bytes(&bytes).map_err(|error| error.to_string())?;
    let (orientation, face_transform) = heif_transform(&bytes)?;
    let handle = context
        .primary_image_handle()
        .map_err(|error| error.to_string())?;
    let profile = handle
        .color_profile_raw()
        .filter(|profile| profile.data.len() <= 16 * 1024 * 1024)
        .map(|profile| HeifColorProfile::Icc(profile.data))
        .or_else(|| handle.color_profile_nclx().map(HeifColorProfile::Nclx));
    let has_alpha = handle.has_alpha_channel();
    let bit_depth = handle.luma_bits_per_pixel();
    if analysis_memory_budget.is_some_and(|budget| {
        !heif_allocation_within_budget(
            bytes.len(),
            handle.width() as usize,
            handle.height() as usize,
            has_alpha,
            bit_depth > 8,
            budget,
        )
    }) {
        return Err("HEIF dimensions exceed the decoder allocation budget".into());
    }
    let color_space = match (bit_depth > 8, has_alpha, cfg!(target_endian = "little")) {
        (false, false, _) => ColorSpace::Rgb(RgbChroma::Rgb),
        (false, true, _) => ColorSpace::Rgb(RgbChroma::Rgba),
        (true, false, true) => ColorSpace::Rgb(RgbChroma::HdrRgbLe),
        (true, false, false) => ColorSpace::Rgb(RgbChroma::HdrRgbBe),
        (true, true, true) => ColorSpace::Rgb(RgbChroma::HdrRgbaLe),
        (true, true, false) => ColorSpace::Rgb(RgbChroma::HdrRgbaBe),
    };
    let decoded = LibHeif::new()
        .decode(&handle, color_space, None)
        .map_err(|error| error.to_string())?;
    let plane = decoded
        .planes()
        .interleaved
        .ok_or_else(|| "decoded HEIF has no interleaved RGB plane".to_owned())?;
    let channels = if has_alpha { 4 } else { 3 };
    let sample_bytes = if bit_depth > 8 { 2 } else { 1 };
    let row_bytes = (plane.width as usize)
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(sample_bytes))
        .ok_or_else(|| "decoded HEIF dimensions overflow".to_owned())?;
    if row_bytes > plane.stride {
        return Err("decoded HEIF RGB row exceeds its plane stride".into());
    }
    let capacity = row_bytes
        .checked_mul(plane.height as usize)
        .ok_or_else(|| "decoded HEIF dimensions overflow".to_owned())?;
    if plane.data.len() < plane.stride.saturating_mul(plane.height as usize) {
        return Err("decoded HEIF plane is truncated".into());
    }
    let mut pixels = Vec::with_capacity(capacity);
    for row in plane
        .data
        .chunks_exact(plane.stride)
        .take(plane.height as usize)
    {
        if cancelled.load(Ordering::Acquire) {
            return Err("render cancelled".into());
        }
        pixels.extend_from_slice(&row[..row_bytes]);
    }
    let image = if sample_bytes == 1 {
        convert_profile_to_srgb_8(&mut pixels, channels, profile.as_ref());
        if has_alpha {
            DynamicImage::ImageRgba8(
                ImageBuffer::<Rgba<u8>, _>::from_raw(plane.width, plane.height, pixels)
                    .ok_or_else(|| "invalid decoded HEIF RGBA buffer".to_owned())?,
            )
        } else {
            DynamicImage::ImageRgb8(
                ImageBuffer::<Rgb<u8>, _>::from_raw(plane.width, plane.height, pixels)
                    .ok_or_else(|| "invalid decoded HEIF RGB buffer".to_owned())?,
            )
        }
    } else {
        let shift = 16_u8.saturating_sub(bit_depth);
        let mut samples: Vec<u16> = pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]) << shift)
            .collect();
        convert_profile_to_srgb_16(&mut samples, channels, profile.as_ref());
        if has_alpha {
            DynamicImage::ImageRgba16(
                ImageBuffer::<Rgba<u16>, _>::from_raw(plane.width, plane.height, samples)
                    .ok_or_else(|| "invalid decoded HEIF 16-bit RGBA buffer".to_owned())?,
            )
        } else {
            DynamicImage::ImageRgb16(
                ImageBuffer::<Rgb<u16>, _>::from_raw(plane.width, plane.height, samples)
                    .ok_or_else(|| "invalid decoded HEIF 16-bit RGB buffer".to_owned())?,
            )
        }
    };
    if cancelled.load(Ordering::Acquire) {
        Err("render cancelled".into())
    } else {
        Ok((image, orientation, face_transform))
    }
}

pub(super) fn heif_compressed_source_admitted(compressed_bytes: u64, budget: usize) -> bool {
    // Before dimensions are available, reserve the fixed allocations known to
    // coexist with the compressed source: maximum accepted ICC and analysis output.
    let fixed = 16_u64 * 1024 * 1024 + 2048_u64 * 2048 * 4;
    compressed_bytes
        .checked_add(fixed)
        .is_some_and(|peak| peak <= budget as u64)
}

pub(super) fn heif_pre_read_admitted(
    compressed_bytes: u64,
    analysis_budget: Option<usize>,
) -> bool {
    analysis_budget.is_none_or(|budget| heif_compressed_source_admitted(compressed_bytes, budget))
}

pub(super) fn heif_allocation_within_budget(
    compressed_bytes: usize,
    width: usize,
    height: usize,
    alpha: bool,
    sixteen_bit: bool,
    budget: usize,
) -> bool {
    heif_peak_bytes(compressed_bytes, width, height, alpha, sixteen_bit)
        .is_some_and(|peak| peak <= budget)
}

pub(super) fn heif_peak_bytes(
    compressed_bytes: usize,
    width: usize,
    height: usize,
    alpha: bool,
    sixteen_bit: bool,
) -> Option<usize> {
    let channels = if alpha { 4_usize } else { 3 };
    let sample_bytes = if sixteen_bit { 2_usize } else { 1 };
    let plane = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .and_then(|samples| samples.checked_mul(sample_bytes))?;
    // Compressed source bytes, libheif plane, copied conversion buffer,
    // DynamicImage ownership and bounded analysis derivative coexist briefly.
    let concurrent_planes = if sixteen_bit { 4 } else { 3 };
    compressed_bytes
        .checked_add(plane.saturating_mul(concurrent_planes))
        .and_then(|peak| peak.checked_add(16 * 1024 * 1024))
        .and_then(|peak| peak.checked_add(2048 * 2048 * 4))
}

fn heif_transform(bytes: &[u8]) -> Result<(u8, crate::rendering::FaceTransform), String> {
    use libheif_sys as sys;
    // SAFETY: bytes outlive the context and handle, and both native objects are
    // released before this function returns.
    unsafe {
        let context = sys::heif_context_alloc();
        if context.is_null() {
            return Err("cannot allocate HEIF context".to_owned());
        }
        let read = sys::heif_context_read_from_memory_without_copy(
            context,
            bytes.as_ptr().cast(),
            bytes.len(),
            std::ptr::null(),
        );
        if read.code != 0 {
            sys::heif_context_free(context);
            return Err("cannot inspect HEIF transformations".to_owned());
        }
        let mut handle = std::ptr::null_mut();
        let primary = sys::heif_context_get_primary_image_handle(context, &mut handle);
        if primary.code != 0 || handle.is_null() {
            sys::heif_context_free(context);
            return Err("cannot inspect primary HEIF image".to_owned());
        }
        let item_id = sys::heif_image_handle_get_item_id(handle);
        let mut width = sys::heif_image_handle_get_ispe_width(handle);
        let mut height = sys::heif_image_handle_get_ispe_height(handle);
        if width <= 0 || height <= 0 {
            sys::heif_image_handle_release(handle);
            sys::heif_context_free(context);
            return Err("invalid HEIF canonical dimensions".to_owned());
        }
        let count =
            sys::heif_item_get_transformation_properties(context, item_id, std::ptr::null_mut(), 0)
                .max(0) as usize;
        let mut properties = vec![0; count];
        if count > 0 {
            sys::heif_item_get_transformation_properties(
                context,
                item_id,
                properties.as_mut_ptr(),
                count as i32,
            );
        }
        let mut orientation = 1_u8;
        let mut face_transform = crate::rendering::FaceTransform::IDENTITY;
        for property in properties {
            let property_type = sys::heif_item_get_property_type(context, item_id, property);
            let operation = if property_type
                == sys::heif_item_property_type_heif_item_property_type_transform_rotation
            {
                match sys::heif_item_get_property_transform_rotation_ccw(context, item_id, property)
                {
                    90 => Some(8),
                    180 => Some(3),
                    270 => Some(6),
                    _ => None,
                }
            } else if property_type
                == sys::heif_item_property_type_heif_item_property_type_transform_mirror
            {
                match sys::heif_item_get_property_transform_mirror(context, item_id, property) {
                    sys::heif_transform_mirror_direction_heif_transform_mirror_direction_horizontal => Some(2),
                    sys::heif_transform_mirror_direction_heif_transform_mirror_direction_vertical => Some(4),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(operation) = operation {
                orientation = crate::rendering::compose_orientations(orientation, operation);
                face_transform =
                    face_transform.then(crate::rendering::FaceTransform::orientation(operation));
                if matches!(operation, 5..=8) {
                    std::mem::swap(&mut width, &mut height);
                }
            } else if property_type
                == sys::heif_item_property_type_heif_item_property_type_transform_crop
            {
                let (mut left, mut top, mut right, mut bottom) = (0, 0, 0, 0);
                sys::heif_item_get_property_transform_crop_borders(
                    context,
                    item_id,
                    property,
                    width,
                    height,
                    &mut left,
                    &mut top,
                    &mut right,
                    &mut bottom,
                );
                if left < 0
                    || top < 0
                    || right < 0
                    || bottom < 0
                    || left + right >= width
                    || top + bottom >= height
                {
                    sys::heif_image_handle_release(handle);
                    sys::heif_context_free(context);
                    return Err("invalid HEIF clean aperture".to_owned());
                }
                face_transform = face_transform.then(crate::rendering::FaceTransform::crop(
                    left as f32 / width as f32,
                    top as f32 / height as f32,
                    right as f32 / width as f32,
                    bottom as f32 / height as f32,
                ));
                width -= left + right;
                height -= top + bottom;
            }
        }
        sys::heif_image_handle_release(handle);
        sys::heif_context_free(context);
        Ok((orientation, face_transform))
    }
}

enum HeifColorProfile {
    Icc(Vec<u8>),
    Nclx(libheif_rs::ColorProfileNCLX),
}

fn mox_profile(profile: &HeifColorProfile) -> Result<moxcms::ColorProfile, String> {
    match profile {
        HeifColorProfile::Icc(data) => {
            moxcms::ColorProfile::new_from_slice(data).map_err(|error| error.to_string())
        }
        HeifColorProfile::Nclx(nclx) => {
            let primaries = moxcms::CicpColorPrimaries::try_from(nclx.color_primaries() as u8)
                .map_err(|error| error.to_string())?;
            let transfer =
                moxcms::TransferCharacteristics::try_from(nclx.transfer_characteristics() as u8)
                    .map_err(|error| error.to_string())?;
            let matrix = moxcms::MatrixCoefficients::try_from(nclx.matrix_coefficients() as u8)
                .map_err(|error| error.to_string())?;
            Ok(moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
                color_primaries: primaries,
                transfer_characteristics: transfer,
                matrix_coefficients: matrix,
                full_range: nclx.full_range_flag() != 0,
            }))
        }
    }
}

fn convert_profile_to_srgb_8(
    pixels: &mut [u8],
    channels: usize,
    profile: Option<&HeifColorProfile>,
) {
    let Some(source) = profile.and_then(|profile| mox_profile(profile).ok()) else {
        return;
    };
    let destination = moxcms::ColorProfile::new_srgb();
    let layout = if channels == 4 {
        moxcms::Layout::Rgba
    } else {
        moxcms::Layout::Rgb
    };
    let Ok(transform) =
        source.create_transform_8bit(layout, &destination, layout, Default::default())
    else {
        return;
    };
    let source_pixels = pixels.to_vec();
    let _ = transform.transform(&source_pixels, pixels);
}

fn convert_profile_to_srgb_16(
    pixels: &mut [u16],
    channels: usize,
    profile: Option<&HeifColorProfile>,
) {
    let Some(source) = profile.and_then(|profile| mox_profile(profile).ok()) else {
        return;
    };
    let destination = moxcms::ColorProfile::new_srgb();
    let layout = if channels == 4 {
        moxcms::Layout::Rgba
    } else {
        moxcms::Layout::Rgb
    };
    let Ok(transform) =
        source.create_transform_16bit(layout, &destination, layout, Default::default())
    else {
        return;
    };
    let source_pixels = pixels.to_vec();
    let _ = transform.transform(&source_pixels, pixels);
}
