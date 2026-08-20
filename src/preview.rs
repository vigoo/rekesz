use std::{collections::HashMap, path::Path, sync::Arc};

use gpui::{RenderImage, SharedString};
use image::{DynamicImage, Frame, ImageDecoder};

use crate::discovery::Catalog;

#[derive(Clone)]
pub struct Preview {
    pub image: Option<Arc<RenderImage>>,
    pub error: Option<SharedString>,
}

/// Decodes each directory's preview at most once and retains the GPUI-ready
/// image, including its applied metadata orientation.
#[derive(Default)]
pub struct PreviewCache {
    entries: HashMap<std::path::PathBuf, Preview>,
}

impl PreviewCache {
    pub fn get(&mut self, directory: &Path, catalog: &Catalog) -> Preview {
        if let Some(preview) = self.entries.get(directory) {
            return preview.clone();
        }

        let preview = decode_first(directory, catalog);
        self.entries
            .insert(directory.to_path_buf(), preview.clone());
        preview
    }
}

fn decode_first(directory: &Path, catalog: &Catalog) -> Preview {
    let Some(entry) = catalog.directory(directory) else {
        return Preview {
            image: None,
            error: Some("Directory is not in the catalog".into()),
        };
    };
    if let Some(error) = &entry.scan_error {
        return Preview {
            image: None,
            error: Some(format!("Cannot read directory: {error}").into()),
        };
    }

    for path in &entry.images {
        match oriented_render_image(path) {
            Ok(image) => {
                return Preview {
                    image: Some(Arc::new(image)),
                    error: None,
                };
            }
            Err(_) => continue,
        }
    }

    Preview {
        image: None,
        error: None,
    }
}

fn oriented_render_image(path: &Path) -> image::ImageResult<RenderImage> {
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
