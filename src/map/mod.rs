use crate::config::OnlineMapPermission;
use crate::metadata::GeoCoordinate;

pub(crate) mod network;
pub(crate) mod renderer;

pub(crate) const TILEJSON_ENDPOINT: &str = "https://tiles.openfreemap.org/planet";
pub(crate) const ATTRIBUTION: &str =
    "Natural Earth · OpenFreeMap © OpenMapTiles · Data from OpenStreetMap contributors";
pub(crate) const USER_AGENT: &str = concat!("Rekesz/", env!("CARGO_PKG_VERSION"));
pub(crate) const BUNDLED_DETAIL_MAX_ZOOM: f64 = 5.0;
pub(crate) const PROVIDER_MAX_ZOOM: f64 = 14.0;

/// Proof that the user has granted online-map access. Network code accepts this
/// capability instead of a boolean so `Unknown` and `Disabled` cannot request even TileJSON.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OnlineMapAccess(());

impl OnlineMapAccess {
    pub fn from_permission(permission: OnlineMapPermission) -> Option<Self> {
        (permission == OnlineMapPermission::Enabled).then_some(Self(()))
    }

    pub fn tilejson_endpoint(self) -> &'static str {
        TILEJSON_ENDPOINT
    }

    pub fn user_agent(self) -> &'static str {
        USER_AGENT
    }

    pub fn accepts_resource(self, resource: &str) -> bool {
        is_allowed_provider_resource(resource)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Camera {
    pub center: GeoCoordinate,
    pub zoom: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: GeoCoordinate::new(20.0, 0.0).expect("default map center is valid"),
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn zoom_by(&mut self, delta: f64) {
        self.zoom = (self.zoom + delta).clamp(0.0, 20.0);
    }

    pub fn fit_viewport(
        coordinates: &[GeoCoordinate],
        viewport: (u32, u32),
        padding: f64,
    ) -> Option<Self> {
        coordinates.first()?;
        let mut longitudes = coordinates
            .iter()
            .map(|coordinate| coordinate.longitude())
            .collect::<Vec<_>>();
        longitudes.sort_by(f64::total_cmp);
        let (west, _, longitude_span) = shortest_longitude_arc(&longitudes);
        let center_longitude = normalize_longitude(west + longitude_span / 2.0);
        let (min_y, max_y) = coordinates
            .iter()
            .map(|coordinate| mercator_project(*coordinate).1)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), y| {
                (min.min(y), max.max(y))
            });
        let center_y = (min_y + max_y) / 2.0;
        let center_latitude = mercator_unproject(0.5, center_y)?.latitude();
        let available_width = (f64::from(viewport.0) - padding * 2.0).max(1.0);
        let available_height = (f64::from(viewport.1) - padding * 2.0).max(1.0);
        let x_span = (longitude_span / 360.0).max(1.0 / 2_f64.powi(18));
        let y_span = (max_y - min_y).max(1.0 / 2_f64.powi(18));
        let zoom_x = (available_width / (256.0 * x_span)).log2();
        let zoom_y = (available_height / (256.0 * y_span)).log2();
        Some(Self {
            center: GeoCoordinate::new(center_latitude, center_longitude)?,
            zoom: zoom_x.min(zoom_y).clamp(0.0, 18.0),
        })
    }

    pub fn visible_bounds(self, viewport: (u32, u32)) -> Vec<[f64; 4]> {
        let (center_x, center_y) = mercator_project(self.center);
        let world = 256.0 * 2_f64.powf(self.zoom);
        let half_x = f64::from(viewport.0) / (2.0 * world);
        let half_y = f64::from(viewport.1) / (2.0 * world);
        let north = mercator_unproject(center_x, center_y - half_y)
            .map_or(85.051_13, GeoCoordinate::latitude);
        let south = mercator_unproject(center_x, center_y + half_y)
            .map_or(-85.051_13, GeoCoordinate::latitude);
        if half_x * 2.0 >= 1.0 {
            return vec![[-180.0, south, 180.0, north]];
        }
        let west_x = center_x - half_x;
        let east_x = center_x + half_x;
        let west = normalize_longitude(west_x * 360.0 - 180.0);
        let east = normalize_longitude(east_x * 360.0 - 180.0);
        if west_x.floor() == east_x.floor() && west <= east {
            vec![[west, south, east, north]]
        } else {
            vec![[-180.0, south, east, north], [west, south, 180.0, north]]
        }
    }
}

fn shortest_longitude_arc(longitudes: &[f64]) -> (f64, f64, f64) {
    if longitudes.len() <= 1 {
        let longitude = longitudes.first().copied().unwrap_or(0.0);
        return (longitude, longitude, 0.0);
    }
    let mut largest_gap = -1.0;
    let mut gap_end = 0;
    for index in 0..longitudes.len() {
        let current = longitudes[index];
        let next = if index + 1 == longitudes.len() {
            longitudes[0] + 360.0
        } else {
            longitudes[index + 1]
        };
        if next - current > largest_gap {
            largest_gap = next - current;
            gap_end = index;
        }
    }
    let west = longitudes[(gap_end + 1) % longitudes.len()];
    let span = 360.0 - largest_gap;
    (west, normalize_longitude(west + span), span)
}

fn normalize_longitude(longitude: f64) -> f64 {
    (longitude + 180.0).rem_euclid(360.0) - 180.0
}

pub(crate) fn mercator_project(coordinate: GeoCoordinate) -> (f64, f64) {
    let latitude = coordinate
        .latitude()
        .clamp(-85.051_13, 85.051_13)
        .to_radians();
    (
        (coordinate.longitude() + 180.0) / 360.0,
        (1.0 - latitude.tan().asinh() / std::f64::consts::PI) / 2.0,
    )
}

pub(crate) fn mercator_unproject(x: f64, y: f64) -> Option<GeoCoordinate> {
    let longitude = normalize_longitude(x * 360.0 - 180.0);
    let latitude = (std::f64::consts::PI * (1.0 - 2.0 * y.clamp(0.0, 1.0)))
        .sinh()
        .atan()
        .to_degrees();
    GeoCoordinate::new(latitude, longitude)
}

/// Restricts every renderer request and redirect target to the provider's fixed HTTPS host.
pub(crate) fn is_allowed_provider_resource(resource: &str) -> bool {
    if resource
        .strip_prefix("https://")
        .and_then(|rest| rest.split_once('/'))
        .map(|(authority, _)| authority)
        != Some("tiles.openfreemap.org")
    {
        return false;
    }
    let Ok(url) = reqwest::Url::parse(resource) else {
        return false;
    };
    if url.scheme() != "https"
        || url.host_str() != Some("tiles.openfreemap.org")
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return false;
    }
    let Some(path) = url.path().strip_prefix('/') else {
        return false;
    };
    let lowercase = path.to_ascii_lowercase();
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains(['?', '#', '\\'])
        && !["%2e", "%2f", "%5c", "%25", "%3f", "%23"]
            .iter()
            .any(|encoded| lowercase.contains(encoded))
        && !path
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
        && matches!(
            path.split('/').next(),
            Some("planet" | "fonts" | "sprites" | "natural_earth")
        )
}

pub(crate) fn immutable_tile_template(resource: &str) -> bool {
    if !is_allowed_provider_resource(resource) {
        return false;
    }
    let Some(path) = resource.strip_prefix("https://tiles.openfreemap.org/planet/") else {
        return false;
    };
    let Some((version, template)) = path.split_once('/') else {
        return false;
    };
    version.is_ascii()
        && version.len() >= 8
        && version.as_bytes()[..8].iter().all(u8::is_ascii_digit)
        && version
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        && template == "{z}/{x}/{y}.pbf"
}

pub(crate) struct PhotoClusters {
    index: Option<supercluster::Supercluster>,
    coordinates: Vec<GeoCoordinate>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClusterMarker {
    pub coordinate: GeoCoordinate,
    pub photo_indices: Vec<usize>,
    pub cluster_id: Option<usize>,
    pub expansion_zoom: Option<u8>,
    pub count: usize,
}

impl PhotoClusters {
    pub fn new(coordinates: &[GeoCoordinate]) -> Self {
        if coordinates.len() < 2 {
            return Self {
                index: None,
                coordinates: coordinates.to_vec(),
            };
        }
        let options = supercluster::Supercluster::builder()
            .radius(48.0)
            .extent(512.0)
            .min_points(2)
            // Supercluster expands leaf points one level above its configured
            // maximum, so 19 makes coincident photos individually available at 20.
            .max_zoom(19)
            .coordinate_system(supercluster::CoordinateSystem::LatLng)
            .build();
        let features = supercluster::Supercluster::feature_builder()
            .add_points(
                coordinates
                    .iter()
                    .map(|coordinate| vec![coordinate.longitude(), coordinate.latitude()])
                    .collect(),
            )
            .build();
        let mut index = supercluster::Supercluster::new(options);
        let index = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.load(features)?;
            Ok::<_, supercluster::SuperclusterError>(index)
        }))
        .ok()
        .and_then(Result::ok);
        Self {
            index,
            coordinates: coordinates.to_vec(),
        }
    }

    fn unclustered_markers(&self) -> Vec<ClusterMarker> {
        self.coordinates
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coordinate)| ClusterMarker {
                coordinate,
                photo_indices: vec![index],
                cluster_id: None,
                expansion_zoom: None,
                count: 1,
            })
            .collect()
    }

    pub fn markers(&self, bounds: [f64; 4], zoom: u8) -> Vec<ClusterMarker> {
        let Some(index) = &self.index else {
            return self.unclustered_markers();
        };
        let clustered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.get_clusters(bounds, zoom)
        }));
        let Ok(Ok(features)) = clustered else {
            return self.unclustered_markers();
        };
        let leaf_markers = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            features
                .into_iter()
                .filter_map(|feature| {
                    let geojson::Value::Point(point) = &feature.geometry.as_ref()?.value else {
                        return None;
                    };
                    let coordinate = GeoCoordinate::new(*point.get(1)?, *point.first()?)?;
                    let cluster_id = feature
                        .property("cluster_id")
                        .and_then(serde_json::Value::as_u64)
                        .map(|id| id as usize);
                    let photo_indices = if cluster_id.is_some() {
                        Vec::new()
                    } else {
                        match feature.id.clone()? {
                            geojson::feature::Id::String(id) => vec![id.parse().ok()?],
                            geojson::feature::Id::Number(id) => vec![id.as_u64()? as usize],
                        }
                    };
                    let count = feature
                        .property("point_count")
                        .and_then(serde_json::Value::as_u64)
                        .map_or(photo_indices.len(), |count| count as usize);
                    let expansion_zoom = cluster_id.map(|cluster_id| {
                        index.get_cluster_expansion_zoom(cluster_id).min(20) as u8
                    });
                    Some(ClusterMarker {
                        coordinate,
                        photo_indices,
                        cluster_id,
                        expansion_zoom,
                        count,
                    })
                })
                .collect::<Vec<_>>()
        }));
        let Ok(mut markers) = leaf_markers else {
            return self.unclustered_markers();
        };
        for marker in &mut markers {
            marker.photo_indices.sort_unstable();
            marker.photo_indices.dedup();
        }
        if zoom >= 20 {
            let mut merged = std::collections::BTreeMap::<(u64, u64), ClusterMarker>::new();
            for marker in markers {
                let key = (
                    marker.coordinate.latitude().to_bits(),
                    marker.coordinate.longitude().to_bits(),
                );
                merged
                    .entry(key)
                    .and_modify(|group| group.photo_indices.extend(&marker.photo_indices))
                    .or_insert(marker);
            }
            markers = merged.into_values().collect();
            for marker in &mut markers {
                marker.photo_indices.sort_unstable();
                marker.photo_indices.dedup();
                marker.count = marker.photo_indices.len();
                marker.cluster_id = None;
                marker.expansion_zoom = None;
            }
        }
        markers.retain(|marker| marker.cluster_id.is_some() || !marker.photo_indices.is_empty());
        markers
    }

    pub fn cluster_leaves_page(
        &self,
        cluster_id: usize,
        limit: usize,
        offset: usize,
    ) -> Vec<usize> {
        let Some(index) = &self.index else {
            return Vec::new();
        };
        index
            .get_leaves(cluster_id, limit, offset)
            .into_iter()
            .filter_map(|leaf| match leaf.id? {
                geojson::feature::Id::String(id) => id.parse().ok(),
                geojson::feature::Id::Number(id) => id.as_u64().map(|id| id as usize),
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
