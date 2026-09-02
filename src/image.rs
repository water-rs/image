//! Image views drawn through the engine-neutral scene contract.
//!
//! This module provides [`Image`], a view that displays decoded pixels. The
//! pixels become a [`peniko::ImageBrush`] once, at construction, and are drawn
//! by a single `Scene2D::draw_image` call — so the same view renders on the GPU
//! compute rasterizer, on the CPU sparse-strip rasterizer an embedded build
//! uses, and inside a backend that owns its own scene.
//!
//! # Example
//!
//! ```
//! use waterui_image::{ContentMode, Image};
//!
//! // One red pixel, stretched to cover whatever box the parent offers while
//! // keeping its aspect ratio.
//! let image = Image::new(vec![255, 0, 0, 255], 1, 1)
//!     .resizable()
//!     .content_mode(ContentMode::Fill);
//! ```

use alloc::borrow::ToOwned;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;
use num_traits::ToPrimitive;

use half::f16;
use peniko::color::{AlphaColor, LinearSrgb};
use peniko::{
    Blob, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality, ImageSampler,
};
use waterui_core::{Binding, Environment, SignalExt, View};
use waterui_graphics::{Scene2D, SceneContent, SceneInvalidator, SceneView};
use waterui_layout::{ContentMode, frame::Frame};

use crate::codec::{self, DecodedRgba};
use crate::scene::{self, ImageSceneContent};

pub use crate::codec::DecodePath;

/// An image view.
///
/// `Image` owns its decoded pixels as a shared [`ImageData`] blob and draws
/// them as one scene image command. Placement inside the box the layout gives
/// the view is a transform, not a pipeline: see [`Image::resizable`] and
/// [`Image::content_mode`].
///
/// # Example
///
/// ```
/// use waterui_image::Image;
///
/// // RGBA pixel data (4 bytes per pixel)
/// let pixels: Vec<u8> = vec![255, 0, 0, 255]; // 1x1 red pixel
///
/// let image = Image::new(pixels, 1, 1);
/// assert_eq!(image.dimensions(), (1, 1));
/// ```
#[derive(Debug, Clone)]
pub struct Image {
    brush: ImageBrush,
    /// When `true`, the image takes the box its parent proposes instead of
    /// locking to its native pixel size like a `SwiftUI` `Image` (the default).
    resizable: bool,
    /// Aspect handling inside that box. `None` stretches each axis
    /// independently, which is what `.resizable()` alone means.
    content_mode: Option<ContentMode>,
}

/// How the image should be filtered when its pixel grid does not align
/// with the destination pixel grid (which it almost never does on modern
/// fractional-DPR displays).
///
/// `Linear` is the conventional default for photographs and decoded
/// assets, while `Nearest` preserves sharp pixel edges and is the right
/// choice for icons, pixel art, and rasterized barcodes.
///
/// `#[non_exhaustive]` so future modes (e.g. cubic / Lanczos) can be added
/// without breaking exhaustive `match` statements downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Interpolation {
    /// Bilinear / trilinear sampling. Default for photo-like content.
    #[default]
    Linear,
    /// Nearest-neighbor sampling. Use for pixel art, icons, barcodes.
    Nearest,
}

impl Interpolation {
    /// The sampling quality a scene engine should give this mode.
    ///
    /// `Low` is nearest-neighbour and `Medium` is bilinear in every engine that
    /// implements the contract, which is exactly the two modes offered here.
    const fn to_quality(self) -> ImageQuality {
        match self {
            Self::Linear => ImageQuality::Medium,
            Self::Nearest => ImageQuality::Low,
        }
    }
}

/// Reinhard tone mapping, matching what an HDR source used to be mapped with
/// on its way to a standard-range target.
fn tone_map_reinhard(component: f32) -> f32 {
    let safe = component.max(0.0);
    safe / (safe + 1.0)
}

/// Converts linear `RGBA16F` pixels into the sRGB-encoded 8-bit pixels a scene
/// image brush carries.
///
/// The scene contract's image is 8-bit, so a float source is resolved here
/// rather than by a shader at draw time: tone mapped when it holds
/// high-dynamic-range values, then encoded with the sRGB transfer function that
/// a renderer will decode it with.
fn rgba16f_to_srgb8(pixels: &[u8], high_dynamic_range: bool) -> Vec<u8> {
    pixels
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|texel| {
            let component = |index: usize| {
                let offset = index * 2;
                f32::from(f16::from_le_bytes([texel[offset], texel[offset + 1]]))
            };
            let map = |component: f32| {
                if high_dynamic_range {
                    tone_map_reinhard(component)
                } else {
                    component
                }
            };
            AlphaColor::<LinearSrgb>::new([
                map(component(0)),
                map(component(1)),
                map(component(2)),
                component(3).clamp(0.0, 1.0),
            ])
            .to_rgba8()
            .to_u8_array()
        })
        .collect()
}

/// The byte count `width` x `height` pixels of `format` occupy.
fn size_in_bytes(format: ImageFormat, width: u32, height: u32) -> usize {
    format
        .size_in_bytes(width, height)
        .expect("image dimensions must not overflow a byte count")
}

impl Image {
    /// Creates a new Image from RGBA pixel data.
    ///
    /// The pixel data must be in sRGB-encoded RGBA format (4 bytes per pixel,
    /// straight alpha) and have exactly `width * height * 4` bytes.
    ///
    /// # Arguments
    ///
    /// * `pixels` - RGBA pixel data (4 bytes per pixel)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Panics
    ///
    /// Panics if the pixel data length doesn't match `width * height * 4`.
    #[must_use]
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(
            pixels.len(),
            size_in_bytes(ImageFormat::Rgba8, width, height),
            "Pixel data length must be width * height * 4"
        );
        Self::from_rgba8(pixels, width, height)
    }

    /// Creates a new Image from `RGBA16F` pixel data.
    ///
    /// The pixel data must be in `RGBA16F` format (8 bytes per pixel,
    /// little-endian half-float components holding linear values) and have
    /// exactly `width * height * 8` bytes. High-dynamic-range values are tone
    /// mapped into the standard range on the way in.
    ///
    /// # Panics
    ///
    /// Panics if the pixel data length doesn't match `width * height * 8`.
    #[must_use]
    pub fn new_rgba16f(pixels: &[u8], width: u32, height: u32) -> Self {
        Self::new_rgba16f_with_metadata(pixels, width, height, true)
    }

    #[must_use]
    fn new_rgba16f_with_metadata(
        pixels: &[u8],
        width: u32,
        height: u32,
        high_dynamic_range: bool,
    ) -> Self {
        assert_eq!(
            pixels.len(),
            size_in_bytes(ImageFormat::Rgba8, width, height) * 2,
            "Pixel data length must be width * height * 8 for RGBA16F"
        );
        Self::from_rgba8(rgba16f_to_srgb8(pixels, high_dynamic_range), width, height)
    }

    fn from_rgba8(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            brush: ImageBrush {
                image: ImageData {
                    data: Blob::from(pixels),
                    format: ImageFormat::Rgba8,
                    alpha_type: ImageAlphaType::Alpha,
                    width,
                    height,
                },
                sampler: ImageSampler::default()
                    .with_quality(Interpolation::default().to_quality()),
            },
            resizable: false,
            content_mode: None,
        }
    }

    /// Sets the sampling mode for this image.
    ///
    /// Defaults to [`Interpolation::Linear`]. Switch to
    /// [`Interpolation::Nearest`] for content that should keep crisp pixel
    /// edges across non-integer scale factors (icons, pixel art, rasterized
    /// barcodes).
    #[must_use]
    pub const fn interpolation(mut self, mode: Interpolation) -> Self {
        self.brush.sampler.quality = mode.to_quality();
        self
    }

    /// Allows this image to stretch to its proposed bounds instead of
    /// locking to its native pixel size.
    ///
    /// Mirrors `SwiftUI`'s `Image.resizable()`. The default behaviour
    /// frames the image to its source `width × height` so a 64-pixel
    /// asset stays 64 pixels tall regardless of the parent's proposal;
    /// once `.resizable()` is applied the image fills whatever the
    /// parent gives it, distorting the aspect ratio unless
    /// [`Image::content_mode`] says otherwise.
    #[must_use]
    pub const fn resizable(mut self) -> Self {
        self.resizable = true;
        self
    }

    /// Preserves the aspect ratio inside the box the layout gives this view.
    ///
    /// Mirrors `SwiftUI`'s `.aspectRatio(contentMode:)`:
    /// [`ContentMode::Fit`] scales the image down until it sits entirely
    /// inside the box, centred, and [`ContentMode::Fill`] scales it up until it
    /// covers the box, centred and clipped to it. Without this the image
    /// stretches each axis independently.
    ///
    /// Only meaningful together with [`Image::resizable`]: a non-resizable
    /// image is framed to its own pixel size, where every mode agrees.
    #[must_use]
    pub const fn content_mode(mut self, mode: ContentMode) -> Self {
        self.content_mode = Some(mode);
        self
    }

    /// Get the image dimensions (width, height).
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    /// Get the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.brush.image.width
    }

    /// Get the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.brush.image.height
    }

    /// Decode encoded image bytes and construct an `Image`.
    ///
    /// # Errors
    ///
    /// Returns an error when the encoded image cannot be decoded into drawable pixels.
    pub fn from_encoded(data: &[u8]) -> Result<Self, String> {
        codec::decode_to_rgba8(data).map(Self::from_decoded)
    }

    /// Decode encoded image bytes and report which decode path was selected.
    ///
    /// # Errors
    ///
    /// Returns an error when the encoded image cannot be decoded into drawable pixels.
    pub fn from_encoded_with_path(data: &[u8]) -> Result<(Self, DecodePath), String> {
        codec::decode_to_rgba8_with_path(data)
            .map(|(decoded, path)| (Self::from_decoded(decoded), path))
    }

    /// Build an incremental decoder that accepts binary image stream chunks.
    #[must_use]
    pub fn stream_decoder(content_type: Option<&str>) -> ImageStreamDecoder {
        ImageStreamDecoder::new(content_type)
    }

    /// Renders this image into an offscreen RGBA8 target.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying offscreen render fails.
    #[cfg(feature = "gpu")]
    #[expect(
        clippy::future_not_send,
        reason = "image rendering awaits the UI-local offscreen scene environment"
    )]
    pub async fn render_offscreen(
        self,
        runtime: &waterui_graphics::GpuRuntime,
        config: waterui_graphics::OffscreenRenderConfig,
        env: &mut Environment,
    ) -> Result<waterui_graphics::OffscreenRenderOutput, waterui_graphics::OffscreenRenderError>
    {
        SceneView::new(self.into_scene_content())
            .into_gpu_surface()
            .render_offscreen(runtime, config, env)
            .await
    }

    fn from_decoded(decoded: DecodedRgba) -> Self {
        match decoded.pixel_format {
            waterkit_codec::DecodedPixelFormat::Rgba8UnormSrgb => {
                Self::new(decoded.pixels, decoded.width, decoded.height)
            }
            waterkit_codec::DecodedPixelFormat::Rgba16Float => Self::new_rgba16f_with_metadata(
                &decoded.pixels,
                decoded.width,
                decoded.height,
                decoded.hdr,
            ),
            other => {
                panic!("Image::from_decoded: unsupported decoded pixel format: {other:?}");
            }
        }
    }

    /// The scene content that draws this image, dropping the layout wrapper.
    fn into_scene_content(self) -> ImageSceneContent {
        ImageSceneContent::new(self.brush, self.content_mode)
    }
}

impl View for Image {
    fn body(self, _env: &Environment) -> impl View {
        let width = u32_to_f32(self.width());
        let height = u32_to_f32(self.height());
        let resizable = self.resizable;
        let frame = Frame::new(SceneView::new(self.into_scene_content()));
        if resizable {
            frame
        } else {
            frame.width(width).height(height)
        }
    }
}

/// The state one [`ReactiveImage`] shares with its handle.
struct ReactiveImageState {
    /// The frame currently on display, or `None` while there is nothing to draw.
    brush: RefCell<Option<ImageBrush>>,
    dimensions: Binding<Option<(u32, u32)>>,
    invalidator: RefCell<Option<SceneInvalidator>>,
}

impl fmt::Debug for ReactiveImageState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReactiveImageState")
            .field("dimensions", &self.dimensions.get())
            .finish_non_exhaustive()
    }
}

impl ReactiveImageState {
    fn publish(&self, brush: Option<ImageBrush>) {
        self.dimensions.set(
            brush
                .as_ref()
                .map(|brush| (brush.image.width, brush.image.height)),
        );
        *self.brush.borrow_mut() = brush;
        if let Some(invalidator) = self.invalidator.borrow().as_ref() {
            invalidator();
        }
    }
}

/// A handle that publishes decoded frames into one persistent [`ReactiveImage`].
#[derive(Clone, Debug)]
pub struct ReactiveImageHandle {
    state: Rc<ReactiveImageState>,
}

impl ReactiveImageHandle {
    /// Replaces the displayed frame without replacing the image view.
    ///
    /// Sampling mode travels with the frame; the view's own `resizable` and
    /// content-mode settings are the ones that place it.
    pub fn set(&self, image: Image) {
        self.state.publish(Some(image.brush));
    }

    /// Removes the displayed frame without replacing the image view.
    pub fn clear(&self) {
        self.state.publish(None);
    }
}

/// An image view whose decoded frame can change without rebuilding its subtree.
#[derive(Debug)]
pub struct ReactiveImage {
    state: Rc<ReactiveImageState>,
    resizable: bool,
    content_mode: Option<ContentMode>,
}

impl ReactiveImage {
    /// Allows this image to stretch to its proposed bounds.
    #[must_use]
    pub const fn resizable(mut self) -> Self {
        self.resizable = true;
        self
    }

    /// Preserves the aspect ratio inside the box the layout gives this view.
    ///
    /// See [`Image::content_mode`].
    #[must_use]
    pub const fn content_mode(mut self, mode: ContentMode) -> Self {
        self.content_mode = Some(mode);
        self
    }
}

impl View for ReactiveImage {
    fn body(self, _env: &Environment) -> impl View {
        let width = self
            .state
            .dimensions
            .map(|dimensions| dimensions.map_or(0.0, |(width, _)| u32_to_f32(width)))
            .computed();
        let height = self
            .state
            .dimensions
            .map(|dimensions| dimensions.map_or(0.0, |(_, height)| u32_to_f32(height)))
            .computed();
        let frame = Frame::new(SceneView::new(ReactiveImageSceneContent {
            state: Rc::clone(&self.state),
            content_mode: self.content_mode,
        }));
        if self.resizable {
            frame
        } else {
            frame.width(width).height(height)
        }
    }
}

/// Creates one persistent image view and its precise frame-update handle.
#[must_use]
pub fn reactive_image() -> (ReactiveImageHandle, ReactiveImage) {
    let state = Rc::new(ReactiveImageState {
        brush: RefCell::new(None),
        dimensions: Binding::container(None),
        invalidator: RefCell::new(None),
    });
    (
        ReactiveImageHandle {
            state: Rc::clone(&state),
        },
        ReactiveImage {
            state,
            resizable: false,
            content_mode: None,
        },
    )
}

/// Scene content that draws whichever frame the handle last published.
struct ReactiveImageSceneContent {
    state: Rc<ReactiveImageState>,
    content_mode: Option<ContentMode>,
}

impl fmt::Debug for ReactiveImageSceneContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReactiveImageSceneContent")
            .field("content_mode", &self.content_mode)
            .finish_non_exhaustive()
    }
}

impl SceneContent for ReactiveImageSceneContent {
    fn build_scene(&mut self, target: &mut dyn Scene2D, width: f32, height: f32) -> bool {
        if let Some(brush) = self.state.brush.borrow().as_ref() {
            scene::draw(target, brush, self.content_mode, width, height);
        }
        false
    }

    fn set_invalidator(&mut self, invalidator: Option<SceneInvalidator>) {
        *self.state.invalidator.borrow_mut() = invalidator;
    }
}

impl Drop for ReactiveImageSceneContent {
    fn drop(&mut self) {
        self.state.invalidator.borrow_mut().take();
    }
}

/// Convenience constructor for building an Image view inline.
#[must_use]
pub fn image(pixels: Vec<u8>, width: u32, height: u32) -> Image {
    Image::new(pixels, width, height)
}

/// Incremental encoded image decoder for streaming/progressive display.
#[derive(Debug, Clone)]
pub struct ImageStreamDecoder {
    content_type: Option<String>,
    bytes: Vec<u8>,
    attempts: usize,
    next_attempt_at: usize,
    last_fingerprint: Option<u64>,
}

impl ImageStreamDecoder {
    const FIRST_ATTEMPT_BYTES: usize = 24 * 1024;
    const ATTEMPT_STEP_BYTES: usize = 96 * 1024;
    const MAX_ATTEMPTS: usize = 10;
    const MAX_BUFFER_BYTES: usize = 8 * 1024 * 1024;

    /// Creates a new stream decoder for progressive encoded image bytes.
    #[must_use]
    pub fn new(content_type: Option<&str>) -> Self {
        Self {
            content_type: content_type.map(ToOwned::to_owned),
            bytes: Vec::new(),
            attempts: 0,
            next_attempt_at: Self::FIRST_ATTEMPT_BYTES,
            last_fingerprint: None,
        }
    }

    /// Push a stream chunk and optionally produce a progressive frame.
    #[must_use]
    pub fn push_chunk(&mut self, chunk: &[u8]) -> Option<Image> {
        if chunk.is_empty() {
            return None;
        }
        self.bytes.extend_from_slice(chunk);
        let total_len = self.bytes.len();

        if self.attempts >= Self::MAX_ATTEMPTS
            || total_len < self.next_attempt_at
            || total_len > Self::MAX_BUFFER_BYTES
            || !codec::is_progressive_candidate(self.content_type.as_deref(), &self.bytes)
        {
            return None;
        }

        self.attempts += 1;
        self.next_attempt_at = total_len.saturating_add(Self::ATTEMPT_STEP_BYTES);

        let decoded = codec::decode_progressive_frame(&self.bytes)?;

        let fingerprint = frame_fingerprint(&decoded);
        if self.last_fingerprint == Some(fingerprint) {
            return None;
        }
        self.last_fingerprint = Some(fingerprint);
        Some(Image::from_decoded(decoded))
    }

    /// Finish decoding and produce the final full-quality image.
    ///
    /// # Errors
    ///
    /// Returns an error when nothing was buffered, or when the buffered bytes
    /// cannot be decoded.
    pub fn finish(self) -> Result<Image, String> {
        if self.bytes.is_empty() {
            return Err(String::from("image response body was empty"));
        }
        Image::from_encoded(&self.bytes)
    }
}

fn frame_fingerprint(decoded: &DecodedRgba) -> u64 {
    let len = decoded.pixels.len();
    if len == 0 {
        return 0;
    }
    let first = u64::from(decoded.pixels[0]);
    let mid = u64::from(decoded.pixels[len / 2]);
    let last = u64::from(decoded.pixels[len - 1]);
    (u64::from(decoded.width) << 32)
        ^ u64::from(decoded.height)
        ^ (u64::try_from(len).expect("image fingerprint length must fit in u64") << 8)
        ^ first
        ^ (mid << 16)
        ^ (last << 24)
}

fn u32_to_f32(value: u32) -> f32 {
    value
        .to_f32()
        .expect("image dimensions must be representable as f32")
}

#[cfg(test)]
mod tests {
    use super::{Image, Interpolation, reactive_image, rgba16f_to_srgb8};
    use half::f16;
    use peniko::ImageQuality;

    #[test]
    fn reactive_image_replaces_frame_without_replacing_view() {
        let (handle, _view) = reactive_image();
        handle.set(Image::new(alloc::vec![0, 0, 0, 255], 1, 1));

        assert_eq!(handle.state.dimensions.get(), Some((1, 1)));
        let displayed = {
            let brush = handle.state.brush.borrow();
            let brush = brush.as_ref().expect("published frame must be on display");
            (brush.image.width, brush.image.height)
        };
        assert_eq!(displayed, (1, 1));

        handle.clear();
        assert_eq!(handle.state.dimensions.get(), None);
        assert!(handle.state.brush.borrow().is_none());
    }

    #[test]
    fn interpolation_selects_the_sampling_quality() {
        let image = Image::new(alloc::vec![0, 0, 0, 255], 1, 1);
        assert_eq!(image.brush.sampler.quality, ImageQuality::Medium);
        assert_eq!(
            image
                .interpolation(Interpolation::Nearest)
                .brush
                .sampler
                .quality,
            ImageQuality::Low
        );
    }

    fn half_texel(components: [f32; 4]) -> alloc::vec::Vec<u8> {
        components
            .into_iter()
            .flat_map(|component| f16::from_f32(component).to_le_bytes())
            .collect()
    }

    #[test]
    fn linear_float_pixels_are_srgb_encoded() {
        // Linear 0.5 is sRGB 188, the classic mid-grey check: a pipeline that
        // wrote the linear value straight out would produce 128.
        let converted = rgba16f_to_srgb8(&half_texel([0.5, 0.5, 0.5, 1.0]), false);
        assert_eq!(converted, alloc::vec![188, 188, 188, 255]);
    }

    #[test]
    fn high_dynamic_range_pixels_are_tone_mapped_before_encoding() {
        // Reinhard maps 1.0 to 0.5 linear, which encodes to the same 188.
        let converted = rgba16f_to_srgb8(&half_texel([1.0, 1.0, 1.0, 1.0]), true);
        assert_eq!(converted, alloc::vec![188, 188, 188, 255]);
        // Without tone mapping the same value is full white instead.
        let converted = rgba16f_to_srgb8(&half_texel([1.0, 1.0, 1.0, 1.0]), false);
        assert_eq!(converted, alloc::vec![255, 255, 255, 255]);
    }
}
