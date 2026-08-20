# pics

A small GPUI directory image browser. The left panel contains directories only;
selecting one displays its first decodable image (sorted by filename).

```sh
cargo run --release -- --image-cache-size 8 /path/to/pictures
```

Thumbnails and RAW fit-window previews are persisted in the operating system's
application cache directory (`~/.cache/pics` on Linux). Use
`--cache-root /path/to/cache` to override it. Entries are keyed by source path,
modification time, file size, render dimensions, and rendering recipe; cached
pixels and their transformation metadata are written atomically as PNG/JSON
pairs before being reused by later sessions.

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

Images open fitted to the available window. Double-click the image to switch to
1:1 pixels, then drag with the left mouse button to pan; double-click again to
return to fit-window mode.

Fit-window mode uses a separate on-demand RAW preview cache. Preview dimensions
come from the current display bounds multiplied by its scale factor, so resizing
the window does not trigger regeneration. Moving to a larger or higher-DPI
display upgrades the cached preview; smaller displays reuse the larger result.

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
