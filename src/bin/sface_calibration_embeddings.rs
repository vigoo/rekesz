//! Extracts model-authentic embeddings from the checked-in fictional calibration
//! contact sheets. Output is intentionally stdout-only so the verifier can keep
//! the biometric vectors ephemeral and retain aggregate metrics only.

use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use ort::{session::Session, value::Tensor};
use serde::Serialize;

const YUNET: &[u8] = include_bytes!("../../assets/face/face_detection_yunet_2023mar.onnx");
const SFACE: &[u8] = include_bytes!("../../assets/face/face_recognition_sface_2021dec.onnx");

#[derive(Clone)]
struct Face {
    bounds: [f32; 4],
    landmarks: [[f32; 2]; 5],
    confidence: f32,
}

#[derive(Serialize)]
struct Sample {
    identity: usize,
    variant: usize,
    detector_confidence: f32,
    embedding: Vec<f32>,
}

struct Models {
    detector: Session,
    recognizer: Session,
}

impl Models {
    fn new() -> Result<Self> {
        Ok(Self {
            // Match the production engine's bounded ORT execution policy so
            // calibration measures the same inference configuration and does
            // not create host-sized thread pools.
            detector: Session::builder()?
                .with_intra_threads(2)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .with_inter_threads(1)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .commit_from_memory(YUNET)?,
            recognizer: Session::builder()?
                .with_intra_threads(2)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .with_inter_threads(1)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .commit_from_memory(SFACE)?,
        })
    }

    fn detect(&mut self, image: &DynamicImage) -> Result<Vec<Face>> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let scale = (640.0 / width as f32).min(640.0 / height as f32);
        let resized_width = (width as f32 * scale).round().clamp(1.0, 640.0) as u32;
        let resized_height = (height as f32 * scale).round().clamp(1.0, 640.0) as u32;
        let rgb =
            image::imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle);
        let plane = 640 * 640;
        let mut input = vec![0.0_f32; plane * 3];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let index = y as usize * 640 + x as usize;
            input[index] = f32::from(pixel[2]);
            input[plane + index] = f32::from(pixel[1]);
            input[plane * 2 + index] = f32::from(pixel[0]);
        }
        let tensor = Tensor::from_array(([1_usize, 3, 640, 640], input))?;
        let output = self.detector.run(ort::inputs![tensor])?;
        let names = [
            "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16",
            "bbox_32", "kps_8", "kps_16", "kps_32",
        ];
        let mut values = Vec::with_capacity(names.len());
        for name in names {
            values.push(output[name].try_extract_tensor::<f32>()?.1);
        }
        let mut candidates = Vec::new();
        for (level, stride) in [8_u32, 16, 32].into_iter().enumerate() {
            let columns = 640 / stride;
            let rows = 640 / stride;
            for row in 0..rows {
                for column in 0..columns {
                    let index = (row * columns + column) as usize;
                    let confidence = (values[level][index].clamp(0.0, 1.0)
                        * values[level + 3][index].clamp(0.0, 1.0))
                    .sqrt();
                    if confidence < 0.9 {
                        continue;
                    }
                    let boxes = values[level + 6];
                    let points = values[level + 9];
                    let cx = (column as f32 + boxes[index * 4]) * stride as f32;
                    let cy = (row as f32 + boxes[index * 4 + 1]) * stride as f32;
                    let box_width = boxes[index * 4 + 2].exp() * stride as f32;
                    let box_height = boxes[index * 4 + 3].exp() * stride as f32;
                    let mut landmarks = [[0.0; 2]; 5];
                    for (point_index, point) in landmarks.iter_mut().enumerate() {
                        point[0] =
                            (points[index * 10 + point_index * 2] + column as f32) * stride as f32;
                        point[1] =
                            (points[index * 10 + point_index * 2 + 1] + row as f32) * stride as f32;
                    }
                    candidates.push(Face {
                        bounds: [
                            (cx - box_width / 2.0) / scale / width as f32,
                            (cy - box_height / 2.0) / scale / height as f32,
                            (cx + box_width / 2.0) / scale / width as f32,
                            (cy + box_height / 2.0) / scale / height as f32,
                        ],
                        landmarks: landmarks.map(|point| {
                            [
                                point[0] / scale / width as f32,
                                point[1] / scale / height as f32,
                            ]
                        }),
                        confidence,
                    });
                }
            }
        }
        candidates.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        let mut kept: Vec<Face> = Vec::new();
        for face in candidates {
            if kept
                .iter()
                .all(|other| intersection_over_union(face.bounds, other.bounds) <= 0.3)
            {
                kept.push(face);
            }
        }
        Ok(kept)
    }

    fn embedding(&mut self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let aligned = align_face(image, face)?;
        let plane = 112 * 112;
        let mut input = vec![0.0_f32; plane * 3];
        for (x, y, pixel) in aligned.enumerate_pixels() {
            let index = (y * 112 + x) as usize;
            input[index] = f32::from(pixel[0]);
            input[plane + index] = f32::from(pixel[1]);
            input[plane * 2 + index] = f32::from(pixel[2]);
        }
        let output = self.recognizer.run(ort::inputs![Tensor::from_array((
            [1_usize, 3, 112, 112],
            input,
        ))?])?;
        let mut embedding = output[0].try_extract_tensor::<f32>()?.1.to_vec();
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            bail!("SFace returned an invalid embedding");
        }
        embedding.iter_mut().for_each(|value| *value /= norm);
        Ok(embedding)
    }
}

fn main() -> Result<()> {
    let paths = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        bail!("usage: sface-calibration-embeddings CONTACT_SHEET...");
    }
    let mut models = Models::new()?;
    let mut samples = Vec::with_capacity(paths.len() * 20);
    for (sheet_index, path) in paths.iter().enumerate() {
        let sheet = image::open(path).with_context(|| format!("open {}", path.display()))?;
        let (width, height) = sheet.dimensions();
        for row in 0..3 {
            for column in 0..4 {
                let x0 = column * width / 4;
                let x1 = (column + 1) * width / 4;
                let y0 = row * height / 3;
                let y1 = (row + 1) * height / 3;
                let cell = sheet.crop_imm(x0, y0, x1 - x0, y1 - y0);
                let variants = match row {
                    0 => vec![(0, cell.clone()), (1, cell.brighten(18))],
                    1 => vec![(2, cell.clone()), (3, cell.brighten(-18))],
                    _ => vec![(4, cell)],
                };
                for (variant, image) in variants {
                    let faces = models.detect(&image)?;
                    let face = faces.first().with_context(|| {
                        format!(
                            "no face in sheet {}, row {}, column {}, variant {}",
                            sheet_index + 1,
                            row,
                            column,
                            variant
                        )
                    })?;
                    if faces.len() != 1 {
                        bail!(
                            "expected one face in sheet {}, row {}, column {}, variant {}; found {}",
                            sheet_index + 1,
                            row,
                            column,
                            variant,
                            faces.len()
                        );
                    }
                    samples.push(Sample {
                        identity: sheet_index * 4 + column as usize,
                        variant,
                        detector_confidence: face.confidence,
                        embedding: models.embedding(&image, face)?,
                    });
                }
            }
        }
    }
    serde_json::to_writer(std::io::stdout(), &samples)?;
    Ok(())
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = width * height;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    intersection / (left_area + right_area - intersection).max(f32::EPSILON)
}

fn align_face(image: &DynamicImage, face: &Face) -> Result<RgbImage> {
    const TARGET: [[f32; 2]; 5] = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ];
    let (width, height) = image.dimensions();
    let source = face
        .landmarks
        .map(|point| [point[0] * width as f32, point[1] * height as f32]);
    let transform = similarity_transform(&source, &TARGET)?;
    let determinant = transform[0] * transform[0] + transform[1] * transform[1];
    if determinant <= f32::EPSILON {
        bail!("degenerate face landmark transform");
    }
    let source = image.to_rgb8();
    let mut aligned = RgbImage::new(112, 112);
    for y in 0..112 {
        for x in 0..112 {
            let dx = x as f32 - transform[2];
            let dy = y as f32 - transform[3];
            let source_x = (transform[0] * dx + transform[1] * dy) / determinant;
            let source_y = (-transform[1] * dx + transform[0] * dy) / determinant;
            aligned.put_pixel(x, y, bilinear(&source, source_x, source_y));
        }
    }
    Ok(aligned)
}

fn similarity_transform(source: &[[f32; 2]; 5], target: &[[f32; 2]; 5]) -> Result<[f32; 4]> {
    let source_mean = mean(source);
    let target_mean = mean(target);
    let (mut denominator, mut a, mut b) = (0.0, 0.0, 0.0);
    for (source, target) in source.iter().zip(target) {
        let (sx, sy) = (source[0] - source_mean[0], source[1] - source_mean[1]);
        let (tx, ty) = (target[0] - target_mean[0], target[1] - target_mean[1]);
        denominator += sx * sx + sy * sy;
        a += sx * tx + sy * ty;
        b += sx * ty - sy * tx;
    }
    if denominator <= f32::EPSILON {
        bail!("degenerate face landmarks");
    }
    a /= denominator;
    b /= denominator;
    Ok([
        a,
        b,
        target_mean[0] - a * source_mean[0] + b * source_mean[1],
        target_mean[1] - b * source_mean[0] - a * source_mean[1],
    ])
}

fn mean(points: &[[f32; 2]; 5]) -> [f32; 2] {
    let sum = points.iter().fold([0.0, 0.0], |sum, point| {
        [sum[0] + point[0], sum[1] + point[1]]
    });
    [sum[0] / 5.0, sum[1] / 5.0]
}

fn bilinear(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x >= image.width() as f32 || y >= image.height() as f32 {
        return Rgb([0, 0, 0]);
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = (
        (x0 + 1).min(image.width() - 1),
        (y0 + 1).min(image.height() - 1),
    );
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let mut result = [0_u8; 3];
    for (channel, output) in result.iter_mut().enumerate() {
        let top = f32::from(image.get_pixel(x0, y0)[channel]) * (1.0 - fx)
            + f32::from(image.get_pixel(x1, y0)[channel]) * fx;
        let bottom = f32::from(image.get_pixel(x0, y1)[channel]) * (1.0 - fx)
            + f32::from(image.get_pixel(x1, y1)[channel]) * fx;
        *output = (top * (1.0 - fy) + bottom * fy).round() as u8;
    }
    Rgb(result)
}
