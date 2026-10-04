use std::{path::PathBuf, sync::OnceLock};

use geojson::{FeatureCollection, GeoJson, GeometryValue};
use rayon::prelude::*;
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use super::Camera;
use super::network::NetworkStatsSnapshot;

const COUNTRIES: &str = include_str!("../../assets/map/ne_110m_admin_0_countries.geojson");
const BORDERS: &str = include_str!("../../assets/map/ne_110m_admin_0_boundary_lines_land.geojson");
const COASTLINES: &str = include_str!("../../assets/map/ne_110m_coastline.geojson");
const LAKES: &str = include_str!("../../assets/map/ne_110m_lakes.geojson");
const PLACES: &str = include_str!("../../assets/map/ne_110m_populated_places.geojson");
const STYLE: &str = include_str!("../../assets/map/rekesz-style.json");
const GLYPHS_0_255: &[u8] = include_bytes!("../../assets/map/glyphs/Noto_Sans_Regular/0-255.pbf");
const SPRITE_INDEX: &str = include_str!("../../assets/map/sprites/rekesz.json");
const SPRITE_IMAGE: &[u8] = include_bytes!("../../assets/map/sprites/rekesz.png");
static COUNTRIES_DATA: OnceLock<Result<FeatureCollection, String>> = OnceLock::new();
static BORDERS_DATA: OnceLock<Result<FeatureCollection, String>> = OnceLock::new();
static COASTLINES_DATA: OnceLock<Result<FeatureCollection, String>> = OnceLock::new();
static LAKES_DATA: OnceLock<Result<FeatureCollection, String>> = OnceLock::new();
static PLACES_DATA: OnceLock<Result<FeatureCollection, String>> = OnceLock::new();
static STYLE_DATA: OnceLock<Result<BundledStyle, String>> = OnceLock::new();
static TILE_DOWNLOAD_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

#[derive(serde::Deserialize)]
struct BundledStyle {
    version: u32,
    background: String,
    land: String,
    water: String,
    coastline: LineStyle,
    boundary: LineStyle,
    place: PlaceStyle,
    font_stack: String,
    glyphs: String,
    sprite: String,
}

#[derive(serde::Deserialize)]
struct LineStyle {
    color: String,
    width: f32,
}

#[derive(serde::Deserialize)]
struct PlaceStyle {
    color: String,
    radius: f32,
    maximum_scalerank: i64,
}

pub(crate) struct RenderedMap {
    pub image: image::RgbaImage,
    pub network: NetworkStatsSnapshot,
    pub detail_tiles: usize,
    pub detail_error: Option<String>,
}

struct Projection {
    center: (f64, f64),
    world: f64,
    size: (f64, f64),
}

#[derive(Clone, Copy)]
struct TileId {
    zoom: u8,
    x: u32,
    y: u32,
}

type TileDownload = (
    TileId,
    Result<Option<super::network::CachedResource>, String>,
);

impl Projection {
    fn new(camera: Camera, width: u32, height: u32) -> Self {
        Self {
            center: super::mercator_project(camera.center),
            world: 256.0 * 2_f64.powf(camera.zoom),
            size: (f64::from(width), f64::from(height)),
        }
    }

    fn point(&self, coordinate: &[f64]) -> Option<(f32, f32)> {
        let longitude = *coordinate.first()?;
        let latitude = *coordinate.get(1)?;
        if !longitude.is_finite() || !latitude.is_finite() {
            return None;
        }
        let latitude = latitude.clamp(-85.051_13, 85.051_13).to_radians();
        let x = (longitude + 180.0) / 360.0;
        let y = (1.0 - latitude.tan().asinh() / std::f64::consts::PI) / 2.0;
        let wrapped_x = (x - self.center.0 + 0.5).rem_euclid(1.0) - 0.5;
        Some((
            (self.size.0 / 2.0 + wrapped_x * self.world) as f32,
            (self.size.1 / 2.0 + (y - self.center.1) * self.world) as f32,
        ))
    }
}

fn collection(data: &str, name: &str) -> Result<FeatureCollection, String> {
    match data.parse::<GeoJson>().map_err(|error| error.to_string())? {
        GeoJson::FeatureCollection(value) => Ok(value),
        _ => Err(format!("bundled {name} is not a feature collection")),
    }
}

fn cached_collection(
    cache: &'static OnceLock<Result<FeatureCollection, String>>,
    data: &'static str,
    name: &'static str,
) -> Result<&'static FeatureCollection, String> {
    cache
        .get_or_init(|| collection(data, name))
        .as_ref()
        .map_err(Clone::clone)
}

fn bundled_style() -> Result<&'static BundledStyle, String> {
    STYLE_DATA
        .get_or_init(|| {
            let style: BundledStyle = serde_json::from_str(STYLE).map_err(|e| e.to_string())?;
            if style.version != 1
                || style.font_stack != "Noto Sans Regular"
                || style.glyphs != "glyphs/{fontstack}/{range}.pbf"
                || style.sprite != "sprites/rekesz"
            {
                return Err("unsupported bundled map style resources".to_owned());
            }
            if !style.coastline.width.is_finite()
                || style.coastline.width <= 0.0
                || !style.boundary.width.is_finite()
                || style.boundary.width <= 0.0
                || !style.place.radius.is_finite()
                || style.place.radius <= 0.0
                || style.place.maximum_scalerank < 0
            {
                return Err("invalid numeric value in bundled map style".to_owned());
            }
            for color in [
                &style.background,
                &style.land,
                &style.water,
                &style.coastline.color,
                &style.boundary.color,
                &style.place.color,
            ] {
                style_color(color)?;
            }
            if GLYPHS_0_255.is_empty()
                || serde_json::from_str::<serde_json::Value>(SPRITE_INDEX).is_err()
                || image::load_from_memory_with_format(SPRITE_IMAGE, image::ImageFormat::Png)
                    .is_err()
            {
                return Err("invalid bundled glyph or sprite resource".to_owned());
            }
            Ok(style)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn style_color(value: &str) -> Result<[u8; 4], String> {
    let value = value
        .strip_prefix('#')
        .ok_or_else(|| format!("invalid style color {value}"))?;
    if value.len() != 6 {
        return Err(format!("invalid style color #{value}"));
    }
    let color = u32::from_str_radix(value, 16).map_err(|error| error.to_string())?;
    Ok([
        ((color >> 16) & 0xff) as u8,
        ((color >> 8) & 0xff) as u8,
        (color & 0xff) as u8,
        255,
    ])
}

fn add_line(path: &mut PathBuilder, line: &[geojson::Position], projection: &Projection) {
    let mut started = false;
    let mut previous_x = 0.0_f32;
    for coordinate in line {
        let Some((mut x, y)) = projection.point(coordinate.as_slice()) else {
            continue;
        };
        if started {
            let world = projection.world as f32;
            while x - previous_x > world / 2.0 {
                x -= world;
            }
            while previous_x - x > world / 2.0 {
                x += world;
            }
            path.line_to(x, y)
        } else {
            path.move_to(x, y);
            started = true
        }
        previous_x = x;
    }
}

fn paint(color: [u8; 4]) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(color[0], color[1], color[2], color[3]);
    paint
}

fn visible_tiles(camera: Camera, viewport: (u32, u32), maximum_zoom: u8) -> Vec<TileId> {
    let zoom = (camera.zoom.floor().max(0.0) as u8).min(maximum_zoom);
    let count = 1_i64 << zoom;
    let (center_x, center_y) = super::mercator_project(camera.center);
    let render_world = 256.0 * 2_f64.powf(camera.zoom);
    let source_scale = count as f64;
    let half_x = f64::from(viewport.0) / (2.0 * render_world);
    let half_y = f64::from(viewport.1) / (2.0 * render_world);
    let min_x = ((center_x - half_x) * source_scale).floor() as i64;
    let max_x = ((center_x + half_x) * source_scale).floor() as i64;
    let min_y = (((center_y - half_y) * source_scale).floor() as i64).clamp(0, count - 1);
    let max_y = (((center_y + half_y) * source_scale).floor() as i64).clamp(0, count - 1);
    let mut tiles = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            tiles.push(TileId {
                zoom,
                x: x.rem_euclid(count) as u32,
                y: y as u32,
            });
        }
    }
    tiles
}

fn download_tiles(
    tiles: &[TileId],
    cancelled: &(impl Fn() -> bool + Sync),
    fetch: impl Fn(TileId) -> Result<Option<super::network::CachedResource>, String> + Sync,
) -> Result<Vec<TileDownload>, String> {
    let pool = TILE_DOWNLOAD_POOL
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .thread_name(|index| format!("map-tile-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)?;
    Ok(pool.install(|| {
        tiles
            .par_iter()
            .map(|tile| {
                if cancelled() {
                    (*tile, Err("map render cancelled".to_owned()))
                } else {
                    (*tile, fetch(*tile))
                }
            })
            .collect()
    }))
}

fn tile_point(projection: &Projection, tile: TileId, extent: f64, x: i32, y: i32) -> (f32, f32) {
    let count = f64::from(1_u32 << tile.zoom);
    let world_x = (f64::from(tile.x) + f64::from(x) / extent) / count;
    let world_y = (f64::from(tile.y) + f64::from(y) / extent) / count;
    let wrapped_x = (world_x - projection.center.0 + 0.5).rem_euclid(1.0) - 0.5;
    (
        (projection.size.0 / 2.0 + wrapped_x * projection.world) as f32,
        (projection.size.1 / 2.0 + (world_y - projection.center.1) * projection.world) as f32,
    )
}

fn draw_mvt_line(
    builder: &mut PathBuilder,
    line: &fast_mvt::MvtLineString,
    projection: &Projection,
    tile: TileId,
    extent: f64,
) {
    for (index, coordinate) in line.0.iter().enumerate() {
        let (x, y) = tile_point(projection, tile, extent, coordinate.x, coordinate.y);
        if index == 0 {
            builder.move_to(x, y);
        } else {
            builder.line_to(x, y);
        }
    }
}

fn draw_vector_tile(
    pixmap: &mut Pixmap,
    bytes: &[u8],
    projection: &Projection,
    tile: TileId,
    cancelled: &impl Fn() -> bool,
) -> Result<usize, String> {
    let reader = fast_mvt::MvtReaderRef::new(bytes).map_err(|error| error.to_string())?;
    let mut painted_features = 0;
    for layer in reader.layers() {
        if cancelled() {
            return Err("map render cancelled".to_owned());
        }
        let (color, fill, width) = match layer.name() {
            "water" => ([29, 65, 78, 235], true, 0.0),
            "park" => ([48, 82, 58, 180], true, 0.0),
            "landcover" | "landuse" => ([58, 78, 61, 120], true, 0.0),
            "building" => ([92, 91, 82, 210], true, 0.0),
            "transportation" => ([190, 181, 160, 230], false, 1.2),
            "boundary" => ([150, 155, 151, 210], false, 0.8),
            "waterway" => ([54, 101, 122, 230], false, 1.0),
            _ => continue,
        };
        let extent = f64::from(layer.extent());
        for feature in layer.features() {
            if cancelled() {
                return Err("map render cancelled".to_owned());
            }
            let geometry = feature.geometry().map_err(|error| error.to_string())?;
            let mut builder = PathBuilder::new();
            match &geometry {
                fast_mvt::MvtGeometry::LineString(line) => {
                    draw_mvt_line(&mut builder, line, projection, tile, extent)
                }
                fast_mvt::MvtGeometry::MultiLineString(lines) => {
                    for line in &lines.0 {
                        draw_mvt_line(&mut builder, line, projection, tile, extent);
                    }
                }
                fast_mvt::MvtGeometry::Polygon(polygon) => {
                    draw_mvt_line(&mut builder, polygon.exterior(), projection, tile, extent);
                    builder.close();
                    for ring in polygon.interiors() {
                        draw_mvt_line(&mut builder, ring, projection, tile, extent);
                        builder.close();
                    }
                }
                fast_mvt::MvtGeometry::MultiPolygon(polygons) => {
                    for polygon in &polygons.0 {
                        draw_mvt_line(&mut builder, polygon.exterior(), projection, tile, extent);
                        builder.close();
                        for ring in polygon.interiors() {
                            draw_mvt_line(&mut builder, ring, projection, tile, extent);
                            builder.close();
                        }
                    }
                }
                _ => continue,
            }
            let Some(path) = builder.finish() else {
                continue;
            };
            let paint = paint(color);
            if fill {
                pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::EvenOdd,
                    Transform::identity(),
                    None,
                );
            } else {
                pixmap.stroke_path(
                    &path,
                    &paint,
                    &Stroke {
                        width,
                        ..Default::default()
                    },
                    Transform::identity(),
                    None,
                );
            }
            painted_features += 1;
        }
    }
    Ok(painted_features)
}

fn draw_polygons(
    pixmap: &mut Pixmap,
    data: &FeatureCollection,
    projection: &Projection,
    color: [u8; 4],
) {
    let paint = paint(color);
    for feature in &data.features {
        let Some(geometry) = &feature.geometry else {
            continue;
        };
        let polygons = match &geometry.value {
            GeometryValue::Polygon { coordinates } => std::slice::from_ref(coordinates),
            GeometryValue::MultiPolygon { coordinates } => coordinates.as_slice(),
            _ => continue,
        };
        let mut builder = PathBuilder::new();
        for polygon in polygons {
            for ring in polygon {
                add_line(&mut builder, ring, projection);
                builder.close();
            }
        }
        if let Some(path) = builder.finish() {
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::EvenOdd,
                Transform::identity(),
                None,
            );
        }
    }
}

fn draw_lines(
    pixmap: &mut Pixmap,
    data: &FeatureCollection,
    projection: &Projection,
    color: [u8; 4],
    width: f32,
) {
    let paint = paint(color);
    let stroke = Stroke {
        width,
        ..Stroke::default()
    };
    for feature in &data.features {
        let Some(geometry) = &feature.geometry else {
            continue;
        };
        let mut builder = PathBuilder::new();
        match &geometry.value {
            GeometryValue::LineString { coordinates } => {
                add_line(&mut builder, coordinates, projection)
            }
            GeometryValue::MultiLineString { coordinates: lines } => {
                for line in lines {
                    add_line(&mut builder, line, projection)
                }
            }
            _ => continue,
        }
        if let Some(path) = builder.finish() {
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
}

fn draw_places(
    pixmap: &mut Pixmap,
    data: &FeatureCollection,
    projection: &Projection,
    style: &PlaceStyle,
) -> Result<(), String> {
    let paint = paint(style_color(&style.color)?);
    for feature in &data.features {
        if feature
            .property("SCALERANK")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(i64::MAX)
            > style.maximum_scalerank
        {
            continue;
        }
        let Some(GeometryValue::Point {
            coordinates: coordinate,
        }) = feature.geometry.as_ref().map(|value| &value.value)
        else {
            continue;
        };
        if let Some((x, y)) = projection.point(coordinate.as_slice()) {
            let circle = PathBuilder::from_circle(x, y, style.radius)
                .ok_or_else(|| "invalid place radius in bundled style".to_owned())?;
            pixmap.fill_path(
                &circle,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
    Ok(())
}

pub(crate) fn render_map(
    width: u32,
    height: u32,
    camera: Camera,
    cache_path: PathBuf,
    cache_capacity: u64,
    online_access: Option<super::OnlineMapAccess>,
    cancelled: impl Fn() -> bool + Sync,
) -> Result<RenderedMap, String> {
    let mut network = NetworkStatsSnapshot::default();
    let mut detail_tiles = 0;
    let mut detail_error = None;
    let mut tile_source = None;
    if cancelled() {
        return Err("map render cancelled".to_owned());
    }
    if camera.zoom > super::BUNDLED_DETAIL_MAX_ZOOM {
        match super::network::fetch_tilejson(&cache_path, cache_capacity, online_access) {
            Ok(Some((tilejson, stats))) => {
                network.hits = network.hits.saturating_add(stats.hits);
                network.misses = network.misses.saturating_add(stats.misses);
                network.downloads = network.downloads.saturating_add(stats.downloads);
                network.errors = network.errors.saturating_add(stats.errors);
                network.not_modified = network.not_modified.saturating_add(stats.not_modified);
                tile_source = Some(tilejson);
            }
            Ok(None) => {}
            Err(error) => {
                network.errors = 1;
                detail_error = Some(error);
            }
        }
    }
    let style = bundled_style()?;
    let mut pixmap =
        Pixmap::new(width, height).ok_or_else(|| "invalid map dimensions".to_owned())?;
    let background = style_color(&style.background)?;
    pixmap.fill(tiny_skia::Color::from_rgba8(
        background[0],
        background[1],
        background[2],
        background[3],
    ));
    let projection = Projection::new(camera, width, height);
    draw_polygons(
        &mut pixmap,
        cached_collection(&COUNTRIES_DATA, COUNTRIES, "countries")?,
        &projection,
        style_color(&style.land)?,
    );
    draw_polygons(
        &mut pixmap,
        cached_collection(&LAKES_DATA, LAKES, "lakes")?,
        &projection,
        style_color(&style.water)?,
    );
    draw_lines(
        &mut pixmap,
        cached_collection(&COASTLINES_DATA, COASTLINES, "coastlines")?,
        &projection,
        style_color(&style.coastline.color)?,
        style.coastline.width,
    );
    draw_lines(
        &mut pixmap,
        cached_collection(&BORDERS_DATA, BORDERS, "borders")?,
        &projection,
        style_color(&style.boundary.color)?,
        style.boundary.width,
    );
    if let Some(tilejson) = tile_source {
        let tiles = visible_tiles(camera, (width, height), tilejson.maxzoom);
        let resources = download_tiles(&tiles, &cancelled, |tile| {
            tilejson
                .tile_url(tile.zoom, tile.x, tile.y)
                .and_then(|url| {
                    super::network::load_resource(&cache_path, cache_capacity, &url, online_access)
                })
        })?;
        for (tile, resource) in resources {
            if cancelled() {
                return Err("map render cancelled".to_owned());
            }
            match resource {
                Ok(Some(resource)) => {
                    network.hits = network.hits.saturating_add(resource.stats.hits);
                    network.misses = network.misses.saturating_add(resource.stats.misses);
                    network.downloads = network.downloads.saturating_add(resource.stats.downloads);
                    network.errors = network.errors.saturating_add(resource.stats.errors);
                    network.not_modified = network
                        .not_modified
                        .saturating_add(resource.stats.not_modified);
                    match draw_vector_tile(
                        &mut pixmap,
                        &resource.body,
                        &projection,
                        tile,
                        &cancelled,
                    ) {
                        Ok(painted) if painted > 0 => detail_tiles += 1,
                        Ok(_) => {}
                        Err(error) => {
                            network.errors = network.errors.saturating_add(1);
                            detail_error = Some(error);
                        }
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    network.errors = network.errors.saturating_add(1);
                    detail_error = Some(error);
                }
            }
        }
    }
    draw_places(
        &mut pixmap,
        cached_collection(&PLACES_DATA, PLACES, "places")?,
        &projection,
        &style.place,
    )?;
    let image = image::RgbaImage::from_raw(width, height, pixmap.take())
        .ok_or_else(|| "invalid rendered image".to_owned())?;
    Ok(RenderedMap {
        image,
        network,
        detail_tiles,
        detail_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use test_r::test;

    fn protobuf_varint(mut value: u64, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    fn protobuf_field(tag: u8, body: &[u8], output: &mut Vec<u8>) {
        output.push(tag << 3 | 2);
        protobuf_varint(body.len() as u64, output);
        output.extend_from_slice(body);
    }

    fn diagonal_line_tile(layer_name: &[u8]) -> Vec<u8> {
        let mut geometry = Vec::new();
        for value in [9, 0, 0, 10, 8192, 8192] {
            protobuf_varint(value, &mut geometry);
        }
        let mut feature = vec![3 << 3, 2];
        protobuf_field(4, &geometry, &mut feature);
        let mut layer = Vec::new();
        layer.extend_from_slice(&[15 << 3, 2]);
        protobuf_field(1, layer_name, &mut layer);
        protobuf_field(2, &feature, &mut layer);
        layer.extend_from_slice(&[5 << 3]);
        protobuf_varint(4096, &mut layer);
        let mut tile = Vec::new();
        protobuf_field(3, &layer, &mut tile);
        tile
    }

    #[test]
    fn pure_rust_renderer_smoke_test() {
        let rendered = render_map(320, 200, Camera::default(), PathBuf::new(), 0, None, || {
            false
        })
        .unwrap();
        assert_eq!(rendered.image.dimensions(), (320, 200));
        assert!(
            rendered
                .image
                .pixels()
                .any(|pixel| pixel.0 != [23, 43, 54, 255])
        );
    }

    #[test]
    fn bundled_datasets_parse() {
        let style = bundled_style().unwrap();
        assert_eq!(style.version, 1);
        assert!(style_color(&style.background).is_ok());
        for (name, data) in [
            ("countries", COUNTRIES),
            ("borders", BORDERS),
            ("coastlines", COASTLINES),
            ("lakes", LAKES),
            ("places", PLACES),
        ] {
            assert!(!collection(data, name).unwrap().features.is_empty());
        }
    }

    #[test]
    fn deterministic_vector_tile_changes_the_rendered_surface() {
        let camera = Camera {
            center: crate::metadata::GeoCoordinate::new(0.0, 0.0).unwrap(),
            zoom: 0.0,
        };
        let projection = Projection::new(camera, 256, 256);
        let mut pixmap = Pixmap::new(256, 256).unwrap();
        pixmap.fill(tiny_skia::Color::from_rgba8(23, 43, 54, 255));
        let before = pixmap.data().to_vec();
        let painted = draw_vector_tile(
            &mut pixmap,
            &diagonal_line_tile(b"transportation"),
            &projection,
            TileId {
                zoom: 0,
                x: 0,
                y: 0,
            },
            &|| false,
        )
        .unwrap();
        assert_eq!(painted, 1);
        let changed = pixmap
            .data()
            .iter()
            .zip(before)
            .filter(|(after, before)| **after != *before)
            .count();
        assert!(changed > 500, "only {changed} vector-tile bytes changed");
    }

    #[test]
    fn unsupported_vector_layers_do_not_claim_visible_detail() {
        let camera = Camera::default();
        let projection = Projection::new(camera, 256, 256);
        let mut pixmap = Pixmap::new(256, 256).unwrap();
        assert_eq!(
            draw_vector_tile(
                &mut pixmap,
                &diagonal_line_tile(b"unknown-layer"),
                &projection,
                TileId {
                    zoom: 0,
                    x: 0,
                    y: 0,
                },
                &|| false,
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn cancelled_render_stops_before_cache_or_network_work() {
        let error = render_map(320, 200, Camera::default(), PathBuf::new(), 0, None, || {
            true
        })
        .err()
        .unwrap();
        assert_eq!(error, "map render cancelled");
    }

    #[test]
    fn visible_tiles_wrap_at_the_antimeridian_and_stay_in_latitude_range() {
        let camera = Camera {
            center: crate::metadata::GeoCoordinate::new(85.0, 179.9).unwrap(),
            zoom: 3.5,
        };
        let tiles = visible_tiles(camera, (1200, 800), 14);
        assert!(!tiles.is_empty());
        assert!(tiles.iter().all(|tile| tile.zoom == 3));
        assert!(tiles.iter().all(|tile| tile.x < 8 && tile.y < 8));
        assert!(tiles.iter().any(|tile| tile.x == 0));
        assert!(tiles.iter().any(|tile| tile.x == 7));
    }

    #[test]
    fn tile_downloads_use_bounded_parallelism() {
        let tiles = (0..8)
            .map(|x| TileId { zoom: 4, x, y: 0 })
            .collect::<Vec<_>>();
        let active = AtomicUsize::new(0);
        let maximum = AtomicUsize::new(0);
        let started = std::time::Instant::now();
        let downloaded = download_tiles(&tiles, &|| false, |_| {
            let active_now = active.fetch_add(1, Ordering::AcqRel) + 1;
            maximum.fetch_max(active_now, Ordering::AcqRel);
            std::thread::sleep(std::time::Duration::from_millis(30));
            active.fetch_sub(1, Ordering::AcqRel);
            Ok(None)
        })
        .unwrap();
        assert_eq!(downloaded.len(), tiles.len());
        assert!((2..=4).contains(&maximum.load(Ordering::Acquire)));
        assert!(started.elapsed() < std::time::Duration::from_millis(220));
    }

    #[test]
    fn adjacent_vector_tiles_share_an_exact_screen_seam() {
        let projection = Projection::new(Camera::default(), 800, 600);
        let left = TileId {
            zoom: 2,
            x: 1,
            y: 1,
        };
        let right = TileId {
            zoom: 2,
            x: 2,
            y: 1,
        };
        assert_eq!(
            tile_point(&projection, left, 4096.0, 4096, 2048),
            tile_point(&projection, right, 4096.0, 0, 2048)
        );
    }

    #[test]
    fn malformed_vector_tile_is_rejected_without_modifying_pixels() {
        let projection = Projection::new(Camera::default(), 64, 64);
        let mut pixmap = Pixmap::new(64, 64).unwrap();
        let before = pixmap.data().to_vec();
        assert!(
            draw_vector_tile(
                &mut pixmap,
                b"not an mvt",
                &projection,
                TileId {
                    zoom: 0,
                    x: 0,
                    y: 0,
                },
                &|| false,
            )
            .is_err()
        );
        assert_eq!(pixmap.data(), before);
    }

    #[test]
    fn bundled_renderer_supports_a_large_viewport() {
        let rendered = render_map(
            1920,
            1080,
            Camera::default(),
            PathBuf::new(),
            0,
            None,
            || false,
        )
        .unwrap();
        assert_eq!(rendered.image.dimensions(), (1920, 1080));
        assert_eq!(rendered.image.as_raw().len(), 1920 * 1080 * 4);
    }

    #[test]
    #[ignore = "requires explicit network access to OpenFreeMap"]
    fn online_vector_renderer_and_offline_cache_smoke_test() {
        let directory = tempfile::tempdir().unwrap();
        let cache = directory.path().join("ambient.sqlite3");
        let camera = Camera {
            center: crate::metadata::GeoCoordinate::new(47.4979, 19.0402).unwrap(),
            zoom: 8.0,
        };
        let access = super::super::OnlineMapAccess::from_permission(
            crate::config::OnlineMapPermission::Enabled,
        );
        let online = render_map(
            512,
            384,
            camera,
            cache.clone(),
            64 * 1024 * 1024,
            access,
            || false,
        )
        .unwrap();
        assert!(
            online.network.downloads > 0,
            "network={:?}, detail_error={:?}",
            online.network,
            online.detail_error
        );
        let offline =
            render_map(512, 384, camera, cache, 64 * 1024 * 1024, None, || false).unwrap();
        assert!(offline.network.hits > 0);
        assert_eq!(online.image.dimensions(), offline.image.dimensions());
        assert_eq!(online.image, offline.image);
        let bundled = render_map(512, 384, camera, PathBuf::new(), 0, None, || false).unwrap();
        let changed_pixels = online
            .image
            .pixels()
            .zip(bundled.image.pixels())
            .filter(|(online, bundled)| online != bundled)
            .count();
        eprintln!("online detail changed {changed_pixels} pixels");
        assert!(changed_pixels > 10_000);
    }
}
