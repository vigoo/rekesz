use super::*;
use test_r::test;

#[test]
fn fits_across_the_antimeridian() {
    let camera = Camera::fit_viewport(
        &[
            GeoCoordinate::new(10.0, 179.0).unwrap(),
            GeoCoordinate::new(12.0, -179.0).unwrap(),
        ],
        (800, 600),
        40.0,
    )
    .unwrap();
    assert!(camera.center.longitude().abs() > 179.0);
    assert!(camera.zoom > 6.0);
}

#[test]
fn viewport_fit_accounts_for_aspect_ratio_and_mercator_latitude() {
    let coordinates = [
        GeoCoordinate::new(69.0, 18.0).unwrap(),
        GeoCoordinate::new(70.0, 19.0).unwrap(),
    ];
    let wide = Camera::fit_viewport(&coordinates, (1200, 400), 40.0).unwrap();
    let tall = Camera::fit_viewport(&coordinates, (400, 1200), 40.0).unwrap();
    assert!(wide.zoom < tall.zoom);
    assert!((69.0..=70.0).contains(&wide.center.latitude()));
}

#[test]
fn viewport_fit_contains_global_and_antimeridian_points_with_padding() {
    let viewport = (1200_u32, 400_u32);
    let padding = 48.0;
    for coordinates in [
        vec![
            GeoCoordinate::new(-60.0, -170.0).unwrap(),
            GeoCoordinate::new(60.0, 170.0).unwrap(),
        ],
        vec![
            GeoCoordinate::new(10.0, 179.0).unwrap(),
            GeoCoordinate::new(12.0, -179.0).unwrap(),
        ],
        vec![
            GeoCoordinate::new(84.5, -50.0).unwrap(),
            GeoCoordinate::new(85.0, 50.0).unwrap(),
        ],
        vec![
            GeoCoordinate::new(-85.0, -50.0).unwrap(),
            GeoCoordinate::new(-84.5, 50.0).unwrap(),
        ],
    ] {
        let camera = Camera::fit_viewport(&coordinates, viewport, padding).unwrap();
        let (center_x, center_y) = mercator_project(camera.center);
        let world = 256.0 * 2_f64.powf(camera.zoom);
        for coordinate in coordinates {
            let (x, y) = mercator_project(coordinate);
            let wrapped_x = (x - center_x + 0.5).rem_euclid(1.0) - 0.5;
            let screen_x = f64::from(viewport.0) / 2.0 + wrapped_x * world;
            let screen_y = f64::from(viewport.1) / 2.0 + (y - center_y) * world;
            assert!((padding - 1e-6..=f64::from(viewport.0) - padding + 1e-6).contains(&screen_x));
            assert!((padding - 1e-6..=f64::from(viewport.1) - padding + 1e-6).contains(&screen_y));
        }
    }
}

#[test]
fn rejects_untrusted_resources_and_redirect_targets() {
    assert!(is_allowed_provider_resource(TILEJSON_ENDPOINT));
    assert!(is_allowed_provider_resource(
        "https://tiles.openfreemap.org/fonts/Noto%20Sans/0-255.pbf"
    ));
    assert!(!is_allowed_provider_resource(
        "http://tiles.openfreemap.org/planet"
    ));
    assert!(!is_allowed_provider_resource(
        "https://tiles.openfreemap.org.evil.test/planet"
    ));
    assert!(!is_allowed_provider_resource(
        "https://tiles.openfreemap.org/planet/../secret"
    ));
    assert!(!is_allowed_provider_resource(
        "https://tiles.openfreemap.org/planet/%2e%2e/secret"
    ));
    for resource in [
        "https://tiles.openfreemap.org:443/planet",
        "https://user@tiles.openfreemap.org/planet",
        "https://tiles.openfreemap.org//planet",
        "https://tiles.openfreemap.org/planet/%252e%252e/secret",
        "https://tiles.openfreemap.org/planet/file%3fquery",
        "https://tiles.openfreemap.org/planet?query",
        "https://tiles.openfreemap.org/planet#fragment",
    ] {
        assert!(
            !is_allowed_provider_resource(resource),
            "accepted {resource}"
        );
    }
}

#[test]
fn accepts_only_versioned_vector_tile_templates() {
    assert!(immutable_tile_template(
        "https://tiles.openfreemap.org/planet/20260823_080002_pt/{z}/{x}/{y}.pbf"
    ));
    assert!(!immutable_tile_template(
        "https://tiles.openfreemap.org/planet/{z}/{x}/{y}.pbf"
    ));
    assert!(!immutable_tile_template(
        "https://other.example/planet/20260823/{z}/{x}/{y}.pbf"
    ));
    assert!(!immutable_tile_template(
        "https://tiles.openfreemap.org/planet/éééé/{z}/{x}/{y}.pbf"
    ));
}

#[test]
fn online_endpoint_requires_remembered_consent() {
    assert!(OnlineMapAccess::from_permission(OnlineMapPermission::Unknown).is_none());
    assert!(OnlineMapAccess::from_permission(OnlineMapPermission::Disabled).is_none());
    let access = OnlineMapAccess::from_permission(OnlineMapPermission::Enabled).unwrap();
    assert_eq!(access.tilejson_endpoint(), TILEJSON_ENDPOINT);
}

#[test]
fn clusters_coincident_markers_until_source_max_zoom() {
    let coordinates = vec![
        GeoCoordinate::new(47.5, 19.0).unwrap(),
        GeoCoordinate::new(47.5, 19.0).unwrap(),
        GeoCoordinate::new(-33.9, 151.2).unwrap(),
    ];
    let clusters = PhotoClusters::new(&coordinates);
    let outer = clusters.markers([-180.0, -90.0, 180.0, 90.0], 0);
    let cluster = outer.iter().find(|marker| marker.count >= 2).unwrap();
    let leaves = clusters.cluster_leaves_page(cluster.cluster_id.unwrap(), 10, 0);
    assert!(leaves.contains(&0));
    assert!(leaves.contains(&1));
    let markers = clusters.markers([-180.0, -90.0, 180.0, 90.0], 20);
    assert_eq!(markers.len(), 2);
    assert!(
        markers
            .iter()
            .any(|marker| marker.photo_indices == vec![0, 1])
    );
}

#[test]
fn cluster_leaves_are_retrieved_in_bounded_gallery_pages() {
    let coordinate = GeoCoordinate::new(47.5, 19.0).unwrap();
    let clusters = PhotoClusters::new(&vec![coordinate; 20]);
    let marker = clusters
        .markers([-180.0, -90.0, 180.0, 90.0], 0)
        .into_iter()
        .find(|marker| marker.cluster_id.is_some())
        .unwrap();
    let cluster_id = marker.cluster_id.unwrap();
    assert_eq!(clusters.cluster_leaves_page(cluster_id, 6, 0).len(), 6);
    assert_eq!(clusters.cluster_leaves_page(cluster_id, 6, 18).len(), 2);
}

#[test]
fn empty_and_single_photo_selections_bypass_supercluster() {
    let empty = PhotoClusters::new(&[]);
    assert!(empty.markers([-180.0, -90.0, 180.0, 90.0], 0).is_empty());

    let coordinate = GeoCoordinate::new(47.5, 19.0).unwrap();
    let single = PhotoClusters::new(&[coordinate]);
    assert_eq!(
        single.markers([-180.0, -90.0, 180.0, 90.0], 0),
        vec![ClusterMarker {
            coordinate,
            photo_indices: vec![0],
            cluster_id: None,
            expansion_zoom: None,
            count: 1,
        }]
    );
}

#[test]
fn mercator_projection_round_trips_at_high_latitude() {
    let coordinate = GeoCoordinate::new(78.2232, 15.6469).unwrap();
    let (x, y) = mercator_project(coordinate);
    let restored = mercator_unproject(x, y).unwrap();
    assert!((restored.latitude() - coordinate.latitude()).abs() < 1e-9);
    assert!((restored.longitude() - coordinate.longitude()).abs() < 1e-9);
}

#[test]
fn visible_bounds_are_local_and_split_at_the_antimeridian() {
    let local = Camera {
        center: GeoCoordinate::new(47.5, 19.0).unwrap(),
        zoom: 8.0,
    }
    .visible_bounds((1200, 800));
    assert_eq!(local.len(), 1);
    assert!(local[0][0] < 19.0 && local[0][2] > 19.0);
    let wrapped = Camera {
        center: GeoCoordinate::new(10.0, 179.9).unwrap(),
        zoom: 6.0,
    }
    .visible_bounds((1200, 800));
    assert_eq!(wrapped.len(), 2);
    assert_eq!(wrapped[0][0], -180.0);
    assert_eq!(wrapped[1][2], 180.0);
}

#[test]
fn large_cluster_index_queries_only_the_visible_viewport() {
    let coordinates = (0..10_000)
        .map(|index| {
            let latitude = -80.0 + f64::from(index % 160) + 0.1;
            let longitude = -179.0 + f64::from(index % 358) + 0.1;
            GeoCoordinate::new(latitude, longitude).unwrap()
        })
        .collect::<Vec<_>>();
    let clusters = PhotoClusters::new(&coordinates);
    let markers = clusters.markers([18.0, 47.0, 20.0, 48.0], 8);
    assert!(markers.len() < coordinates.len() / 10);
    assert!(
        markers
            .iter()
            .flat_map(|marker| &marker.photo_indices)
            .all(|index| *index < coordinates.len())
    );
}
