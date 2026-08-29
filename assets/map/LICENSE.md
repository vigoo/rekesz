The bundled overview geography contains the unmodified Natural Earth public-domain
1:110m countries, coastlines, land-boundary lines, lakes, and populated-places GeoJSON datasets
from the official `nvkelso/natural-earth-vector` repository:
https://github.com/nvkelso/natural-earth-vector

Source files and SHA-256 checksums (retrieved 2026-08-28 from the repository's
`master` branch):

- `ne_110m_admin_0_countries.geojson`: `6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f`
- `ne_110m_admin_0_boundary_lines_land.geojson`: `d42479fd79552cca4eec7f85fcdca717a790d29ff06be7676f1af0568c6d3f7c`
- `ne_110m_coastline.geojson`: `851f581ff5ffb844deed8ae1a9ce22e3c4bb3d74fa342cadb5d8e39b41ae7c3c`
- `ne_110m_lakes.geojson`: `eb02ecc86c82004fccbf979058bfabbbd6c2d07968c7844d38eb1c9152d2ffc9`
- `ne_110m_populated_places.geojson`: `a86028b083182b68c7620fc6e1a8a47ee547cb9cd2fb62ccbb78bea786440899`

Each source URL is `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/<filename>`.

Natural Earth data is in the public domain. No permission is needed to use it.

Map rendering uses the pure-Rust tiny-skia rasterizer (MIT/Apache-2.0) and the
bundled versioned Rekesz style in `rekesz-style.json`. OpenFreeMap detail uses
OpenMapTiles data derived from OpenStreetMap and must
display the attribution included in the application. Provider terms:
https://openfreemap.org/tos/ . OpenStreetMap data is offered under ODbL:
https://www.openstreetmap.org/copyright . OpenMapTiles licensing information:
https://openmaptiles.org/about/

The bundled `Noto Sans Regular` glyph range is served by OpenFreeMap and is
derived from Noto Sans. Noto Sans is licensed under the SIL Open Font License;
the complete license text is packaged as `OFL-NotoSans.txt`. The bundled sprite
atlas is the OpenFreeMap `ofm_f384` atlas retrieved from the MIT-licensed
`hyperknot/openfreemap-styles` project and served at
`https://tiles.openfreemap.org/sprites/ofm_f384/ofm` on 2026-08-28.

- `glyphs/Noto_Sans_Regular/0-255.pbf`: `ef1f38a3f1978591e846e9eaddf8a54f7047f546fc6aaed7872cc53151a5de78`
- `sprites/rekesz.json`: `73e75e58d8c7bb62cc25d9d150660500552f4d04f6eac5efa4e236076773c356`
- `sprites/rekesz.png`: `8996a519d218dc5f98015267709dae272a77bb74ef0ecc5a0992dcf276c1be4c`
