use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::UNIX_EPOCH,
};

use image::{DynamicImage, ImageFormat};

use super::{PhotographicSettings, RenderMetadata, RenderRequest, RenderedImage};

const CACHE_VERSION: &str = "render-v1";

#[derive(Clone, Debug)]
pub struct DiskRenderCache {
    root: PathBuf,
}

impl DiskRenderCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn load(
        &self,
        request: &RenderRequest,
        settings: &PhotographicSettings,
    ) -> Result<Option<RenderedImage>, String> {
        let Some(key) = cache_key(request, settings)? else {
            return Ok(None);
        };
        let image_path = self.root.join(format!("{key}.png"));
        let metadata_path = self.root.join(format!("{key}.json"));
        if !image_path.is_file() || !metadata_path.is_file() {
            return Ok(None);
        }
        let metadata: RenderMetadata = serde_json::from_reader(BufReader::new(
            File::open(metadata_path).map_err(|error| error.to_string())?,
        ))
        .map_err(|error| error.to_string())?;
        let image = image::ImageReader::open(image_path)
            .map_err(|error| error.to_string())?
            .with_guessed_format()
            .map_err(|error| error.to_string())?
            .decode()
            .map_err(|error| error.to_string())?;
        Ok(Some(RenderedImage { image, metadata }))
    }

    pub fn store(
        &self,
        request: &RenderRequest,
        settings: &PhotographicSettings,
        rendered: &RenderedImage,
    ) -> Result<(), String> {
        let Some(key) = cache_key(request, settings)? else {
            return Ok(());
        };
        fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
        let image_path = self.root.join(format!("{key}.png"));
        let metadata_path = self.root.join(format!("{key}.json"));

        let mut image_temp =
            tempfile::NamedTempFile::new_in(&self.root).map_err(|error| error.to_string())?;
        {
            let mut writer = BufWriter::new(image_temp.as_file_mut());
            rendered
                .image
                .write_to(&mut writer, ImageFormat::Png)
                .map_err(|error| error.to_string())?;
            writer.flush().map_err(|error| error.to_string())?;
        }
        image_temp
            .as_file_mut()
            .flush()
            .map_err(|error| error.to_string())?;
        image_temp
            .persist(&image_path)
            .map_err(|error| error.error.to_string())?;

        let mut metadata_temp =
            tempfile::NamedTempFile::new_in(&self.root).map_err(|error| error.to_string())?;
        {
            let mut writer = BufWriter::new(metadata_temp.as_file_mut());
            serde_json::to_writer(&mut writer, &rendered.metadata)
                .map_err(|error| error.to_string())?;
            writer.flush().map_err(|error| error.to_string())?;
        }
        metadata_temp
            .as_file_mut()
            .flush()
            .map_err(|error| error.to_string())?;
        metadata_temp
            .persist(metadata_path)
            .map_err(|error| error.error.to_string())?;
        Ok(())
    }
}

fn cache_key(
    request: &RenderRequest,
    settings: &PhotographicSettings,
) -> Result<Option<String>, String> {
    let source = match fs::metadata(&request.path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let modified = source
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_VERSION.as_bytes());
    hasher.update(request.path.as_os_str().as_encoded_bytes());
    hasher.update(&source.len().to_le_bytes());
    hasher.update(&modified.as_secs().to_le_bytes());
    hasher.update(&modified.subsec_nanos().to_le_bytes());
    hasher.update(
        &serde_json::to_vec(&(request.variant, settings)).map_err(|error| error.to_string())?,
    );
    Ok(Some(hasher.finalize().to_hex().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn stores_and_loads_an_image_with_its_render_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.jpg");
        fs::write(&source, b"source identity").unwrap();
        let cache = DiskRenderCache::new(directory.path().join("cache"));
        let request = RenderRequest::new(
            source,
            super::super::RenderVariant::Thumbnail { width: 300 },
        );
        let rendered = RenderedImage {
            image: DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([1, 2, 3]))),
            metadata: RenderMetadata {
                source_pipeline: super::super::SourcePipeline::Standard,
                photographic: None,
                source_dimensions: (2, 1),
                output_dimensions: (2, 1),
                variant: request.variant,
                orientation_applied: None,
            },
        };
        let settings = PhotographicSettings::default();
        cache.store(&request, &settings, &rendered).unwrap();
        let loaded = cache.load(&request, &settings).unwrap().unwrap();
        assert_eq!(loaded.image.width(), 2);
        assert_eq!(loaded.metadata, rendered.metadata);
    }
}
