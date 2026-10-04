# waterui-image

High-performance image primitives and decode pipeline for WaterUI.

Decoded pixels are drawn through `waterui-graphics`' render-target-neutral
recording contract: one `Draw::image` call at the destination rectangle that
resolves the view's content mode, recorded through `waterui_graphics::draw`.
The same component therefore renders on the GPU compute renderer, the CPU
sparse-strip renderer used on embedded targets, and any backend that owns its
own scene.

Rasterizing an image into a surface of its own — `Image::render_offscreen` —
needs a GPU device, so it sits behind the default-on `gpu` feature. A consumer
whose backend owns the scene turns it off and links no `cherenkov-gpu`, no
`wgpu` and no rasterizer.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
