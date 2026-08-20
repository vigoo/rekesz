# pics

A small GPUI directory image browser. The left panel contains directories only;
selecting one displays its first decodable image (sorted by filename).

```sh
cargo run --release -- /path/to/pictures
```

When the path is omitted, `pics` uses the current directory. Directory symlinks
are intentionally not traversed. Supported formats include AVIF, BMP, DDS, EXR,
Farbfeld, GIF, HDR, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF, and WebP (subject to
the platform renderer's support).

EXIF orientation metadata is applied before the image is handed to GPUI, including
rotated and mirrored orientations supported by the underlying decoder.

The filesystem is indexed eagerly at startup with one `read_dir` call per
directory. The UI renders from that immutable catalog, and decoded previews are
cached after first use, so expanding, collapsing, and revisiting directories do
not rescan the filesystem or decode the same preview again.
