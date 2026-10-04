//! Image views drawn through the engine-neutral content contract.
//!
//! This module provides [`Image`], a view that displays decoded pixels. The
//! pixels register with the recording's [`RecordingResources`] on the first frame
//! `build_scene` draws them, and each recording is a single `Draw::image`
//! call with the destination rectangle placement resolved, so the same view
//! renders on the GPU rasterizer, on a CPU rasterizer an embedded build
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
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use half::f16;
use num_traits::ToPrimitive as _;
use waterui_core::layout::Size as LayoutSize;
use waterui_core::{Binding, Environment, Signal, SignalExt, View};
#[cfg(feature = "gpu")]
use waterui_graphics::cherenkov_gpu::Gpu;
use waterui_graphics::color::linear_to_srgb;
use waterui_graphics::draw::{ImageId, Recorder, Sampling};
#[cfg(feature = "gpu")]
use waterui_graphics::{OffscreenError, OffscreenImage, OffscreenRenderer, OffscreenSize};
use waterui_graphics::{RecordingResources, Registered, SceneContent, SceneInvalidator, SceneView};
use waterui_layout::{ContentMode, frame::Frame};

use crate::codec::{self, DecodedRgba};
use crate::scene::{ImageSceneContent, Pixels, pixel_size, u32_to_f32};

pub use crate::codec::DecodePath;

/// An image view.
///
/// `Image` owns its decoded pixels as a shared [`Pixels`] grid and draws them
/// as one image command. Placement inside the box the layout gives the view is
/// a destination rectangle, not a pipeline: see [`Image::resizable`] and
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
    pixels: Pixels,
    /// How the engine samples between texels.
    sampling: Sampling,
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
    /// The engine sampling mode this interpolation selects.
    const fn to_sampling(self) -> Sampling {
        match self {
            Self::Linear => Sampling::Linear,
            Self::Nearest => Sampling::Nearest,
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
/// image carries.
///
/// The content contract's image is 8-bit, so a float source is resolved here
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
            // The sRGB OETF over the clamped value, quantized to a byte —
            // `AlphaColor::<LinearSrgb>::to_rgba8`, inlined. Alpha carries no
            // transfer encoding.
            let byte = |channel: f32| {
                (channel.clamp(0.0, 1.0) * 255.0)
                    .round()
                    .to_u8()
                    .expect("a clamped channel fits a u8")
            };
            [
                byte(linear_to_srgb(map(component(0)))),
                byte(linear_to_srgb(map(component(1)))),
                byte(linear_to_srgb(map(component(2)))),
                byte(component(3)),
            ]
        })
        .collect()
}

/// The byte count a `width` x `height` grid of `bytes_per_pixel` occupies.
fn size_in_bytes(bytes_per_pixel: usize, width: u32, height: u32) -> usize {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
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
            size_in_bytes(4, width, height),
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
            size_in_bytes(8, width, height),
            "Pixel data length must be width * height * 8 for RGBA16F"
        );
        Self::from_rgba8(rgba16f_to_srgb8(pixels, high_dynamic_range), width, height)
    }

    fn from_rgba8(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            pixels: Pixels::new(pixels, width, height),
            sampling: Interpolation::default().to_sampling(),
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
        self.sampling = mode.to_sampling();
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
        self.pixels.width
    }

    /// Get the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.pixels.height
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

    /// Renders this image into an offscreen target and reads the frame back.
    ///
    /// `renderer` owns the engine the pixels register against; `size` is the
    /// render target in pixels and `scale` the point-to-pixel factor the
    /// content's box derives from — `1.0` draws one point per pixel.
    ///
    /// # Errors
    ///
    /// Returns an error when the offscreen surface cannot be created, the
    /// frame cannot render, or the target cannot be read back.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn render_offscreen(
        self,
        renderer: &OffscreenRenderer<Gpu>,
        size: OffscreenSize,
        scale: f32,
    ) -> Result<OffscreenImage, OffscreenError> {
        renderer.render(&mut self.into_scene_content(), size, scale)
    }

    /// Renders this image into an offscreen target and reads the frame back.
    ///
    /// `renderer` owns the engine the pixels register against; `size` is the
    /// render target in pixels and `scale` the point-to-pixel factor the
    /// content's box derives from — `1.0` draws one point per pixel.
    ///
    /// # Errors
    ///
    /// Returns an error when the offscreen surface cannot be created, the
    /// frame cannot render, or the target cannot be read back.
    #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
    #[allow(
        clippy::future_not_send,
        reason = "the engine's wasm32 API is !Send by design and every future executes on the browser's single-threaded executor"
    )]
    pub async fn render_offscreen(
        self,
        renderer: &OffscreenRenderer<Gpu>,
        size: OffscreenSize,
        scale: f32,
    ) -> Result<OffscreenImage, OffscreenError> {
        renderer
            .render(&mut self.into_scene_content(), size, scale)
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
        ImageSceneContent::new(self.pixels, self.sampling, self.content_mode)
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
    /// The frame last published — pixels plus their sampling — or `None`
    /// while there is nothing to draw. `build_scene` draws from it.
    frame: RefCell<Option<(Pixels, Sampling)>>,
    /// The displayed frame's pixel dimensions, as a binding the view's
    /// frame modifiers read so a size change re-lays out without a rebuild.
    dimensions: Binding<Option<(u32, u32)>>,
    /// The invalidator the mounted content registered, so `publish` can ask
    /// for the frame that re-records the new frame's placement.
    invalidator: RefCell<Option<SceneInvalidator>>,
}

impl fmt::Debug for ReactiveImageState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReactiveImageState")
            .field("dimensions", &self.dimensions.snapshot())
            .field("published", &self.frame.borrow().is_some())
            .finish_non_exhaustive()
    }
}

impl ReactiveImageState {
    /// Stores `frame` as the one to draw and asks the content for the frame
    /// that re-records it.
    ///
    /// Registration itself can only happen inside `build_scene` — that is
    /// where the recording's `RecordingResources` arrive — so a publish
    /// never touches the engine itself. The frame waits in the shared state,
    /// and the next `build_scene` registers its pixels and names the
    /// registration in the recording it produces: a recording already
    /// installed keeps drawing the pixels it was recorded with until that
    /// replacement lands, so no frame ever samples a partly written image.
    fn publish(state: &Rc<Self>, frame: Option<(Pixels, Sampling)>) {
        state.dimensions.set(
            frame
                .as_ref()
                .map(|(pixels, _)| (pixels.width, pixels.height)),
        );
        *state.frame.borrow_mut() = frame;
        if let Some(invalidator) = state.invalidator.borrow().as_ref() {
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
        ReactiveImageState::publish(&self.state, Some((image.pixels, image.sampling)));
    }

    /// Removes the displayed frame without replacing the image view.
    pub fn clear(&self) {
        ReactiveImageState::publish(&self.state, None);
    }
}

/// An image view whose decoded frame can change without rebuilding its
/// subtree.
///
/// One [`SceneView`] mounts [`ReactiveImageSceneContent`] for the view's
/// whole life: a published frame registers through the recording that next
/// builds it, so nothing remounts and no state inside the subtree is lost.
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
            image: None,
            uploaded: None,
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
        frame: RefCell::new(None),
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
///
/// The content holds the [`Registered`] handle for the frame it last
/// uploaded. A publish swaps nothing behind a recording: the next
/// `build_scene` registers the new pixels, and the recording it produces
/// names them. Handles are only ever *released* inside `build_scene` (or on
/// drop, when the content's recording is tearing down anyway): the contract
/// is that a recording still naming a resource is never drawn after its
/// release, and only `build_scene` knows the installed recording has moved
/// on.
struct ReactiveImageSceneContent {
    state: Rc<ReactiveImageState>,
    content_mode: Option<ContentMode>,
    /// The registration of the frame `uploaded` holds, minted lazily inside
    /// `build_scene`. `None` until the first frame that draws pixels.
    image: Option<Registered<ImageId>>,
    /// The pixels `image` carries, so a re-recording of an unchanged frame
    /// keeps naming the one registration instead of re-uploading it.
    uploaded: Option<Pixels>,
}

impl ReactiveImageSceneContent {
    /// Lets go of the registration the recordings have stopped naming.
    fn release(&mut self) {
        self.image = None;
        self.uploaded = None;
    }
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
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        let frame = self.state.frame.borrow().clone();
        let Some((pixels, sampling)) = frame else {
            self.release();
            return false;
        };
        let Some(data) = pixels.upload() else {
            self.release();
            return false;
        };
        // Every publish lands here as a new frame: its pixels get their own
        // registration rather than swapping behind the id the installed
        // recording already names — `Registered` is opaque, and the
        // recording that draws it holds it.
        let unchanged = self.uploaded.as_ref().is_some_and(|uploaded| {
            uploaded.width == pixels.width
                && uploaded.height == pixels.height
                && Arc::ptr_eq(&uploaded.data, &pixels.data)
        });
        if !unchanged {
            self.image =
                Some(resources.image(data).unwrap_or_else(|error| {
                    panic!("image upload rejected by the engine: {error}")
                }));
            self.uploaded = Some(pixels.clone());
        }
        let id = resources.name(
            self.image
                .as_ref()
                .expect("a published frame always has a registration after the upload block"),
        );
        crate::scene::draw(
            recorder,
            id,
            (pixels.width, pixels.height),
            sampling,
            self.content_mode,
            width,
            height,
        );
        false
    }

    // The registration and the `uploaded` frame it was minted from belong to
    // the old engine. The published frame stays in the shared `state` — the
    // semantic source — so the next `build_scene` on the replacement engine
    // registers and draws it like any other publish.
    fn rebuild_for_engine(&mut self) {
        self.release();
    }

    /// The last published frame's pixel grid, and `None` before the first
    /// frame arrives: until then the view has no picture, and so no size of
    /// its own.
    fn intrinsic_size(&self) -> Option<LayoutSize> {
        self.state
            .dimensions
            .snapshot()
            .and_then(|(width, height)| pixel_size(width, height))
    }

    fn set_invalidator(&mut self, invalidator: Option<SceneInvalidator>) {
        *self.state.invalidator.borrow_mut() = invalidator;
    }
}

impl Drop for ReactiveImageSceneContent {
    fn drop(&mut self) {
        self.state.invalidator.borrow_mut().take();
        self.release();
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

#[cfg(test)]
mod tests {
    use super::{
        Image, Interpolation, ReactiveImageSceneContent, ReactiveImageState, reactive_image,
        rgba16f_to_srgb8,
    };
    use alloc::rc::Rc;
    use alloc::vec;
    use core::cell::Cell;
    use half::f16;
    use waterui_core::Signal;
    use waterui_graphics::draw::Sampling;

    #[test]
    fn reactive_image_publishes_the_latest_frame() {
        let (handle, _view) = reactive_image();
        assert!(handle.state.frame.borrow().is_none());

        handle.set(Image::new(vec![0, 0, 0, 255], 1, 1));
        let frame = handle.state.frame.borrow();
        let (pixels, sampling) = frame.as_ref().expect("published frame must be on display");
        assert_eq!((pixels.width, pixels.height), (1, 1));
        assert_eq!(*sampling, Sampling::Linear);
        assert_eq!(handle.state.dimensions.snapshot(), Some((1, 1)));
        drop(frame);

        handle.clear();
        assert!(handle.state.frame.borrow().is_none());
        assert_eq!(handle.state.dimensions.snapshot(), None);
    }

    /// One mounted reactive image over a `Gpu` engine: the registration the
    /// content mints on the first drawn frame, and the image ids each
    /// installed recording names.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    struct Mount {
        engine: Rc<waterui_graphics::cherenkov::Engine<waterui_graphics::cherenkov_gpu::Gpu>>,
        resources: waterui_graphics::SceneResources,
        surface: waterui_graphics::cherenkov::Surface<waterui_graphics::cherenkov_gpu::Gpu>,
        content: ReactiveImageSceneContent,
        invalidations: Rc<Cell<u32>>,
        /// The resources the installed recording holds. A host keeps the
        /// last recording's `HeldResources` until the replacement installs;
        /// dropping it earlier would release every handle it names.
        installed: waterui_graphics::HeldResources,
    }

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    impl Mount {
        fn new(state: Rc<ReactiveImageState>) -> Self {
            use waterui_graphics::cherenkov::{Engine, Offscreen, OffscreenFormat};
            use waterui_graphics::cherenkov_gpu::{Gpu, GpuConfig};
            use waterui_graphics::{
                HeldResources, SceneContent as _, SceneInvalidator, SceneResources,
            };

            let engine = Rc::new(
                Engine::<Gpu>::new(GpuConfig::default()).expect("the GPU engine failed to start"),
            );
            let surface = engine
                .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
                .expect("surface");
            let invalidations = Rc::new(Cell::new(0));
            let mut content = ReactiveImageSceneContent {
                state,
                content_mode: None,
                image: None,
                uploaded: None,
            };
            let observed = Rc::clone(&invalidations);
            let invalidator: SceneInvalidator = Rc::new(move || {
                observed.set(observed.get() + 1);
            });
            content.set_invalidator(Some(invalidator));
            Self {
                resources: SceneResources::new(Rc::clone(&engine)),
                engine,
                surface,
                content,
                invalidations,
                installed: HeldResources::empty(),
            }
        }

        /// Record, install, render: the image ids the recording named.
        fn frame(&mut self) -> alloc::vec::Vec<waterui_graphics::draw::ImageId> {
            use waterui_graphics::SceneContent as _;
            use waterui_graphics::cherenkov::FrameTime;
            use waterui_graphics::draw::Command;

            let mut resources = self.resources.recording();
            let mut recorded = self.surface.record(|recorder| {
                self.content
                    .build_scene(recorder, &mut resources, 64.0, 64.0);
            });
            let held = resources.finish();
            let drawn = recorded
                .snapshot()
                .commands()
                .iter()
                .filter_map(|command| match command {
                    Command::Image { image, .. } => Some(*image),
                    _ => None,
                })
                .collect();
            self.surface.update(|tx| {
                tx[self.surface.root()].content(recorded);
            });
            self.installed = held;
            self.engine.render(FrameTime::now()).expect("render");
            drawn
        }
    }

    /// publish → the next recording registers the new frame → a new
    /// `ImageId`; an unchanged frame keeps naming the one registration, and
    /// clear releases the content's handle inside `build_scene`.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[test]
    fn published_frames_draw_through_their_own_registrations() {
        let (handle, view) = reactive_image();
        let mut mount = Mount::new(Rc::clone(&view.state));

        // First publish before any frame: nothing registered yet — the first
        // build_scene mints it.
        handle.set(Image::new(vec![255, 0, 0, 255], 1, 1));
        assert_eq!(mount.invalidations.get(), 1);
        let drawn = mount.frame();
        let [first] = drawn[..] else {
            panic!("the first drawn frame must name exactly one image: {drawn:?}")
        };
        assert!(
            mount.content.image.is_some(),
            "the mount keeps the registration it recorded"
        );

        // Re-recording the same frame names the same id — the registration
        // is still live, so no second upload happens.
        let drawn = mount.frame();
        assert_eq!(drawn, vec![first], "an unchanged frame names the same id");

        // A publish stores new pixels for build_scene: the re-recorded frame
        // names the new registration, a different id than the old pixels'.
        handle.set(Image::new(
            vec![
                0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255,
            ],
            2,
            2,
        ));
        assert_eq!(mount.invalidations.get(), 2);
        assert_eq!(
            mount
                .content
                .uploaded
                .as_ref()
                .map(|pixels| (pixels.width, pixels.height)),
            Some((1, 1)),
            "publish itself registers nothing: the pixels wait for build_scene"
        );
        let drawn = mount.frame();
        let [second] = drawn[..] else {
            panic!("the published frame must name exactly one image: {drawn:?}")
        };
        assert_ne!(first, second, "new pixels register as a new image");
        assert_eq!(
            mount
                .content
                .uploaded
                .as_ref()
                .map(|pixels| (pixels.width, pixels.height)),
            Some((2, 2)),
            "build_scene registered the published pixels"
        );

        // Clearing releases the registration inside build_scene: the new
        // recording names nothing and the content holds no handle.
        handle.clear();
        let drawn = mount.frame();
        assert_eq!(drawn, []);
        assert!(mount.content.image.is_none());
        assert!(mount.content.uploaded.is_none());
    }

    #[test]
    fn interpolation_selects_the_sampling() {
        let image = Image::new(alloc::vec![0, 0, 0, 255], 1, 1);
        assert_eq!(image.sampling, Sampling::Linear);
        assert_eq!(
            image.interpolation(Interpolation::Nearest).sampling,
            Sampling::Nearest
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
