# pics

A small GPUI directory image browser. The left panel contains directories only;
selecting one displays its first decodable image (sorted by filename).

```sh
cargo run --release -- --image-cache-size 8 /path/to/pictures
```

When a directory contains matching `name.CR3` and `name.JPG` files, the grid
shows one item and uses CR3 by default. Pass `--cr3-jpeg-preference jpeg` to use
the JPEG instead.

When the path is omitted, `pics` uses the current directory. Directory symlinks
are intentionally not traversed. Supported formats include AVIF, BMP, DDS, EXR,
Farbfeld, GIF, HDR, ICO, JPEG, PNG, PNM, QOI, TGA, TIFF, and WebP (subject to
the platform renderer's support), plus Nikon NEF and Canon CR3 RAW files and
iPhone HEIC/HEIF images.

RAW files are developed into scene-linear sRGB before a camera-independent,
histogram-based auto-exposure and filmic tone-mapping stage. Rendering settings
are isolated from decoding and UI state so they can later be included in
thumbnail and filesystem-cache keys.

EXIF orientation metadata is applied before the image is handed to GPUI, including
rotated and mirrored orientations supported by the underlying decoder. RAW
orientation is read from rawler's EXIF metadata and applied before resizing.

Selecting a directory shows its images as thumbnails. Click to select, use
Ctrl-click (Command-click on macOS) to toggle items, and Shift-click to select a
range. Double-click a thumbnail or press Enter to open it; press Escape or use
the Back button to return to the directory.

Directories are discovered lazily and cached after their first request. All
directory reads and image decoding run on GPUI's background executor. The tree
uses GPUI's virtual `uniform_list`, and collapsing a branch removes all of its
descendants from the visible row projection. Revisiting a directory does not
rescan it or decode the same preview again.

A progress bar at the bottom tracks active directory scans and image/thumbnail
renders, and disappears when the current batch has completed.

Successfully decoded images use a bounded least-recently-used cache. The
default holds 8 images and can be changed with `--image-cache-size N`. Evicted
images are decoded again if revisited; cheap missing/error results remain cached.

Drag the divider beside the directory tree to resize it. After clicking the
tree, use Up/Down to move selection, Left/Right to collapse, expand, or enter a
branch, and Enter to activate the current directory. The preview header shows
the displayed image's filename.
