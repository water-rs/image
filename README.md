# waterui-image

High-performance image primitives and decode pipeline for WaterUI.

Decoded pixels are recorded as Cherenkov content through `waterui-graphics`'
`SceneContent` contract: the engine uploads the pixels once, and each record is
one image command into the rectangle the view's content mode resolves. The same
component therefore renders on every Cherenkov backend and inside any backend
that merges the content into a scene of its own.

Rasterizing an image into a surface of its own — `waterui-graphics`'
`OffscreenRenderer` over `Image::into_scene_content()`, and the fallback a
`SceneView` takes when the backend does not merge scenes itself — needs a GPU
device, so it sits behind the default-on `gpu` feature. A consumer whose backend
owns the scene turns it off and links no wgpu.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
