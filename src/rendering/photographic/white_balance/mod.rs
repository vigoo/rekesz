use crate::catalog::{DevelopRecipe, WhiteBalance};

pub(super) fn normalized_coefficients(raw: &rawler::RawImage, recipe: &DevelopRecipe) -> [f32; 4] {
    if let Some(white_balance) = recipe.white_balance {
        absolute_coefficients(raw, white_balance)
    } else {
        coefficients(raw.wb_coeffs, raw.neutralwb(), recipe)
    }
}

fn coefficients(as_shot: [f32; 4], neutral: [f32; 4], recipe: &DevelopRecipe) -> [f32; 4] {
    let mut coefficients = if recipe.white_balance.is_some() {
        // Absolute Lightroom WB values describe a chromaticity, not generic
        // output-RGB gains. The caller replaces this result on the RawImage,
        // so the camera matrix is required to translate that chromaticity into
        // sensor-space multipliers. `neutral` is retained as a safe fallback
        // for invalid or unavailable camera calibration data.
        neutral
    } else {
        as_shot
    };
    if let Some(increment) = recipe.white_balance_increment {
        let temperature = (increment.temperature / 100.0).clamp(-1.0, 1.0);
        let tint = (increment.tint / 100.0).clamp(-1.0, 1.0);
        coefficients[0] *= 1.0 + temperature * 0.15 + tint * 0.05;
        coefficients[1] *= 1.0 - tint * 0.08;
        coefficients[2] *= 1.0 - temperature * 0.15 + tint * 0.05;
    }
    coefficients
}

pub(super) fn absolute_coefficients(
    raw: &rawler::RawImage,
    white_balance: WhiteBalance,
) -> [f32; 4] {
    let (x, y) = tinted_white_point(white_balance.temperature, white_balance.tint);
    // Lightroom's tint scale is proprietary. Treat it as a small displacement
    // perpendicular to the daylight/Planckian locus: positive values move the
    // requested white point away from green and towards magenta.
    let xyz = [x / y, 1.0, (1.0 - x - y) / y];
    let matrix = nearest_color_matrix(raw, white_balance.temperature);
    let mut coefficients = [f32::NAN; 4];
    let components = matrix.as_ref().map_or(0, |matrix| matrix.len() / 3);
    for channel in 0..components.min(4) {
        let row = matrix
            .as_ref()
            .map(|matrix| &matrix[channel * 3..channel * 3 + 3])
            .expect("a selected matrix has this row");
        let response = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2];
        if !response.is_finite() || response <= 0.0 {
            return raw.wb_coeffs;
        }
        coefficients[channel] = response.recip();
    }
    let green = coefficients[1];
    if !green.is_finite() || green <= 0.0 {
        return raw.wb_coeffs;
    }
    for coefficient in &mut coefficients[..components.min(4)] {
        *coefficient /= green;
    }
    coefficients
}

fn nearest_color_matrix(raw: &rawler::RawImage, temperature: f32) -> Option<Vec<f32>> {
    use rawler::imgop::xyz::Illuminant;
    let temperature_for = |illuminant| match illuminant {
        Illuminant::A | Illuminant::Tungsten | Illuminant::IsoStudioTungsten => 2_856.0,
        Illuminant::D50 => 5_003.0,
        Illuminant::D55 => 5_503.0,
        Illuminant::D65 | Illuminant::Daylight | Illuminant::Flash => 6_504.0,
        Illuminant::D75 => 7_504.0,
        _ => 6_000.0,
    };
    raw.color_matrix
        .iter()
        .filter(|(_, matrix)| matches!(matrix.len(), 9 | 12))
        .min_by(|(left, _), (right, _)| {
            (temperature_for(**left) - temperature)
                .abs()
                .total_cmp(&(temperature_for(**right) - temperature).abs())
        })
        .map(|(_, matrix)| matrix.clone())
}

fn tinted_white_point(temperature: f32, tint: f32) -> (f32, f32) {
    let (x, y) = planckian_locus_xy_kang2002(temperature);
    let to_uv = |x: f32, y: f32| {
        let denominator = -2.0 * x + 12.0 * y + 3.0;
        (4.0 * x / denominator, 6.0 * y / denominator)
    };
    let (u, v) = to_uv(x, y);
    let delta = (temperature * 0.01).max(1.0);
    let (u0, v0) = to_uv(
        planckian_locus_xy_kang2002(temperature - delta).0,
        planckian_locus_xy_kang2002(temperature - delta).1,
    );
    let (u1, v1) = to_uv(
        planckian_locus_xy_kang2002(temperature + delta).0,
        planckian_locus_xy_kang2002(temperature + delta).1,
    );
    let length = ((u1 - u0).powi(2) + (v1 - v0).powi(2)).sqrt().max(1e-6);
    let amount = tint.clamp(-150.0, 150.0) * 0.00005;
    let u = u - (v1 - v0) / length * amount;
    let v = v + (u1 - u0) / length * amount;
    let denominator = 2.0 * u - 8.0 * v + 4.0;
    (3.0 * u / denominator, 2.0 * v / denominator)
}

/// Evaluates the piecewise polynomial approximation of the CIE 1931 xy
/// Planckian locus from:
///
/// B. Kang, O. Moon, C. Hong, H. Lee, B. Cho, and Y. Kim, "Design of
/// Advanced Color-Temperature Control System for HDTV Applications",
/// Journal of the Korean Physical Society 41(6), pp. 865-871, 2002.
/// https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART000987865
///
/// Independent reference implementation:
/// https://colour.readthedocs.io/en/master/_modules/colour/temperature/kang2002.html
///
/// The published fit is valid from 1667 K through 25000 K. This function
/// clamps inputs to that interval; it does not estimate CCT from arbitrary xy.
fn planckian_locus_xy_kang2002(temperature: f32) -> (f32, f32) {
    let temperature = temperature.clamp(1_667.0, 25_000.0);
    let t2 = temperature * temperature;
    let t3 = t2 * temperature;
    let x = if temperature <= 4_000.0 {
        -0.266_123_9e9 / t3 - 2.343_58e5 / t2 + 0.877_695_6e3 / temperature + 0.179_910
    } else {
        -3.025_846_9e9 / t3 + 2.107_038e6 / t2 + 0.222_634_7e3 / temperature + 0.240_390
    };
    let y = if temperature <= 2_222.0 {
        -1.106_381_4 * x.powi(3) - 1.348_110_2 * x.powi(2) + 2.185_558_3 * x - 0.202_196_83
    } else if temperature <= 4_000.0 {
        -0.954_947_6 * x.powi(3) - 1.374_185_9 * x.powi(2) + 2.091_37 * x - 0.167_488_67
    } else {
        3.081_758 * x.powi(3) - 5.873_387 * x.powi(2) + 3.751_129_9 * x - 0.370_014_83
    };
    (x, y)
}

#[cfg(test)]
mod tests;
