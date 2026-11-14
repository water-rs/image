# waterui-image

High-performance image primitives and decode pipeline for WaterUI.

Decoded pixels are drawn on a `waterui_graphics::canvas::Canvas`: one `image`
command whose destination rectangle follows the canvas size and resolves the
view's content mode. Whichever renderer draws the screen around the view draws
the image, and what it made of the pixels — ready, or refused as too large for
the device — is the image's `status()`.

Rasterizing an image into a surface of its own — `Image::render_offscreen` —
needs a GPU device, so it sits behind the default-on `gpu` feature. A consumer
whose backend owns the rendering turns it off and links no `cherenkov-gpu`, no
`wgpu` and no rasterizer.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
