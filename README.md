# waterui-image

High-performance image primitives and decode pipeline for WaterUI.

Decoded pixels are drawn through `waterui-graphics`' engine-neutral `Scene2D`
contract: one `draw_image` call under the transform that resolves the view's
content mode. The same component therefore renders on the GPU compute renderer,
the CPU sparse-strip renderer used on embedded targets, and any backend that
owns its own scene.

Rasterizing an image into a surface of its own — `Image::render_offscreen`, and
the fallback a `SceneView` takes when the backend does not merge scenes itself —
needs a GPU device, so it sits behind the default-on `gpu` feature. A consumer
whose backend owns the scene turns it off and links no Vello and no rasterizer.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
