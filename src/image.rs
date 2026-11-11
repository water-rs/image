//! Image views drawn on a `Canvas`.
//!
//! This module provides [`Image`], a view that displays decoded pixels. The
//! pixels are an `ImageSource` the render target registers on first use, and
//! the view is a single `image` command whose destination rectangle follows
//! the canvas size, so whichever renderer draws the screen around the view
//! draws the image.
//!
//! The render target can refuse the pixels — an image beyond its image
//! limits, or one its image budget cannot hold beside the images already
//! registered. A refused image draws nothing and reports the refusal as its
//! status: [`Image::status`] for a static image,
//! [`ReactiveImageHandle::status`] for a reactive one.
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
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use half::f16;
use num_traits::ToPrimitive as _;
use waterui_core::layout::Size as LayoutSize;
use waterui_core::layout::{
    LayoutError, LayoutInvalidationCallback, PlacementInput, Point, ProposalSize, Rect,
    StretchAxis, UnitPoint,
};
use waterui_core::reactive::watcher::BoxWatcherGuard;
use waterui_core::{
    AnyView, Binding, Computed, Environment, Signal, SignalExt, View, flatten_signal,
};
#[cfg(feature = "gpu")]
use waterui_graphics::canvas::Drawing;
use waterui_graphics::canvas::{Canvas, ImageStatus, Sampling};
#[cfg(feature = "gpu")]
use waterui_graphics::cherenkov_gpu::Gpu;
use waterui_graphics::color::linear_to_srgb;
#[cfg(feature = "gpu")]
use waterui_graphics::{OffscreenError, OffscreenImage, OffscreenRenderer, OffscreenSize};
use waterui_layout::{
    ContentMode, Layout, SubView, SubviewPlacement, container::FixedContainer, frame,
};

use crate::codec::{self, DecodedRgba};
#[cfg(feature = "gpu")]
use crate::placement::u32_to_f32;
use crate::placement::{Frame, pixel_size};

pub use crate::codec::DecodePath;

/// An image view.
///
/// `Image` owns its decoded pixels as a shared pixel grid and draws them as one
/// image command. Placement inside the box the layout gives the view is a
/// destination rectangle, not a pipeline: see [`Image::resizable`] and
/// [`Image::content_mode`].
///
/// The render target may refuse to register the pixels; the image then draws
/// nothing and [`Image::status`] reports the refusal.
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
    /// The pixels and how they are sampled between texels.
    frame: Frame,
    /// When `true`, the image takes the box its parent proposes instead of
    /// keeping its natural size and only scaling down (the default).
    resizable: bool,
    /// Aspect handling inside that box. `None` stretches each axis
    /// independently, which is what `.resizable()` alone means.
    content_mode: Option<ContentMode>,
}

/// Why [`Image::render_offscreen`] produced no frame.
#[cfg(feature = "gpu")]
#[derive(Debug, thiserror::Error)]
pub enum RenderOffscreenError {
    /// The offscreen target could not be created, the frame could not
    /// render, or the target could not be read back.
    #[error(transparent)]
    Offscreen(#[from] OffscreenError),
    /// The renderer refused the image's pixels, so the frame would have held
    /// nothing of it. The refusal is the one [`ImageStatus::Refused`] carries.
    #[error("the renderer refused the image: {0}")]
    Rejected(alloc::sync::Arc<dyn core::error::Error + Send + Sync>),
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
    /// Panics if the pixel data length doesn't match `width * height * 4`, or
    /// if the image has no area.
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
            frame: Frame::new(
                pixels,
                width,
                height,
                Interpolation::default().to_sampling(),
            ),
            resizable: false,
            content_mode: None,
        }
    }

    /// What the render target made of this image's pixels.
    ///
    /// [`ImageStatus::Pending`] until a target first draws the image, then
    /// [`ImageStatus::Ready`] — the pixels are on screen — or
    /// [`ImageStatus::Refused`] with the reason the target rejected them: they
    /// exceed its image limits, or its image budget cannot hold them beside
    /// the images already registered. A refused image draws nothing and takes
    /// the layout space its pixel grid asks for.
    ///
    /// Clones of an image share the one pixel grid and so the one status.
    #[must_use]
    pub fn status(&self) -> Computed<ImageStatus> {
        self.frame.status()
    }

    /// Sets the sampling mode for this image.
    ///
    /// Defaults to [`Interpolation::Linear`]. Switch to
    /// [`Interpolation::Nearest`] for content that should keep crisp pixel
    /// edges across non-integer scale factors (icons, pixel art, rasterized
    /// barcodes).
    #[must_use]
    pub fn interpolation(mut self, mode: Interpolation) -> Self {
        self.frame = self.frame.sampled(mode.to_sampling());
        self
    }

    /// Allows this image to fill its proposed bounds instead of only scaling down.
    ///
    /// By default, an unspecified proposal receives the natural pixel size.
    /// A smaller finite proposal scales the image down, preserving its aspect
    /// ratio; a larger proposal never scales it up. The image claims no leftover
    /// space. Unlike a non-resizable `SwiftUI` image, it fits a smaller offer.
    /// Once `.resizable()` is applied the image fills whatever the parent gives
    /// it, distorting the aspect ratio unless
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
    /// image always preserves its aspect ratio and only scales down.
    #[must_use]
    pub const fn content_mode(mut self, mode: ContentMode) -> Self {
        self.content_mode = Some(mode);
        self
    }

    /// Get the image dimensions (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    /// Get the image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.frame.width()
    }

    /// Get the image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.frame.height()
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
    /// image's box derives from — `1.0` draws one point per pixel.
    ///
    /// # Errors
    ///
    /// [`RenderOffscreenError::Offscreen`] when the offscreen surface cannot
    /// be created, the frame cannot render, or the target cannot be read
    /// back; [`RenderOffscreenError::Rejected`] when the renderer's engine
    /// refuses the image's pixels.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn render_offscreen(
        self,
        renderer: &OffscreenRenderer<Gpu>,
        size: OffscreenSize,
        scale: f32,
    ) -> Result<OffscreenImage, RenderOffscreenError> {
        let drawing = self.drawing(size, scale);
        let image = renderer.render_drawing(&drawing, size)?;
        self.rendered(image)
    }

    /// Renders this image into an offscreen target and reads the frame back.
    ///
    /// `renderer` owns the engine the pixels register against; `size` is the
    /// render target in pixels and `scale` the point-to-pixel factor the
    /// image's box derives from — `1.0` draws one point per pixel.
    ///
    /// # Errors
    ///
    /// [`RenderOffscreenError::Offscreen`] when the offscreen surface cannot
    /// be created, the frame cannot render, or the target cannot be read
    /// back; [`RenderOffscreenError::Rejected`] when the renderer's engine
    /// refuses the image's pixels.
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
    ) -> Result<OffscreenImage, RenderOffscreenError> {
        let drawing = self.drawing(size, scale);
        let image = renderer.render_drawing(&drawing, size).await?;
        self.rendered(image)
    }

    /// The image placed in the box `size` pixels make at `scale` pixels per
    /// point, as a standalone recording.
    #[cfg(feature = "gpu")]
    fn drawing(&self, size: OffscreenSize, scale: f32) -> Drawing {
        let bounds = LayoutSize::new(
            u32_to_f32(size.width()) / scale,
            u32_to_f32(size.height()) / scale,
        );
        Drawing::record(bounds, |cx| {
            self.frame.record(cx, bounds, self.content_mode);
        })
    }

    /// `image`, unless the renderer refused the pixels it was rendered from —
    /// the frame then holds nothing of the image.
    #[cfg(feature = "gpu")]
    fn rendered(&self, image: OffscreenImage) -> Result<OffscreenImage, RenderOffscreenError> {
        match self.status().snapshot() {
            ImageStatus::Refused(error) => Err(RenderOffscreenError::Rejected(error)),
            _ => Ok(image),
        }
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
}

impl View for Image {
    fn body(self, _env: &Environment) -> impl View {
        let dimensions = Computed::constant(Some(self.dimensions()));
        let resizable = self.resizable;
        let mode = self.content_mode;
        let natural = Computed::constant(self.frame.size());
        let frame = self.frame;
        let canvas = Canvas::new(move |cx| frame.draw(cx, mode)).intrinsic_size(natural);
        sized(canvas, resizable, dimensions)
    }
}

/// `canvas` laid out as an image: filling the box it is given when
/// `resizable`, otherwise at its pixel size, scaled down to fit a smaller
/// offer.
fn sized(canvas: Canvas, resizable: bool, dimensions: Computed<Option<(u32, u32)>>) -> AnyView {
    if resizable {
        AnyView::new(frame::Frame::new(canvas))
    } else {
        AnyView::new(FixedContainer::new(ImageLayout { dimensions }, (canvas,)))
    }
}

#[derive(Debug)]
struct ImageLayout {
    dimensions: Computed<Option<(u32, u32)>>,
}

impl ImageLayout {
    fn resolve(&self, proposal: ProposalSize) -> LayoutSize {
        let Some(natural) = self
            .dimensions
            .snapshot()
            .and_then(|(width, height)| pixel_size(width, height))
        else {
            return LayoutSize::zero();
        };
        let width_scale = proposal
            .width
            .filter(|width| width.is_finite())
            .map_or(1.0, |width| width / natural.width);
        let height_scale = proposal
            .height
            .filter(|height| height.is_finite())
            .map_or(1.0, |height| height / natural.height);
        let scale = width_scale.min(height_scale).clamp(0.0, 1.0);
        LayoutSize::new(natural.width * scale, natural.height * scale)
    }
}

impl Layout for ImageLayout {
    fn size_that_fits(&self, proposal: ProposalSize, _children: &[&dyn SubView]) -> LayoutSize {
        self.resolve(proposal)
    }

    /// The image box centred in the bounds, where `place` draws it: the leaf
    /// rule.
    fn resolve_extent(
        &self,
        input: PlacementInput,
        _children: &[&dyn SubView],
    ) -> Result<Rect, LayoutError> {
        Ok(input.bounds.align(input.extent()?, UnitPoint::CENTER))
    }

    fn place(
        &self,
        bounds: Rect,
        _proposal: ProposalSize,
        children: &[&dyn SubView],
    ) -> Vec<SubviewPlacement> {
        if children.is_empty() {
            return vec![];
        }
        let size = self.resolve(ProposalSize::new(bounds.width(), bounds.height()));
        let origin = Point::new(
            (bounds.width() - size.width).mul_add(0.5, bounds.x()),
            (bounds.height() - size.height).mul_add(0.5, bounds.y()),
        );
        vec![SubviewPlacement::new(
            Rect::new(origin, size),
            ProposalSize::new(size.width, size.height),
        )]
    }

    fn stretch_axis(&self, _children: &[StretchAxis]) -> StretchAxis {
        StretchAxis::None
    }

    fn watch_invalidation(&self, invalidate: LayoutInvalidationCallback) -> Vec<BoxWatcherGuard> {
        vec![self.dimensions.watch(move |_| invalidate())]
    }
}

/// A handle that publishes decoded frames into one persistent [`ReactiveImage`].
#[derive(Clone, Debug)]
pub struct ReactiveImageHandle {
    /// The frame last published, or `None` while there is nothing to draw. The
    /// view's canvas, its intrinsic size and the handle's status all follow it.
    frame: Binding<Option<Frame>>,
}

impl ReactiveImageHandle {
    /// Replaces the displayed frame without replacing the image view.
    ///
    /// Sampling mode travels with the frame; the view's own `resizable` and
    /// content-mode settings are the ones that place it.
    pub fn set(&self, image: Image) {
        self.frame.set(Some(image.frame));
    }

    /// Removes the displayed frame without replacing the image view.
    pub fn clear(&self) {
        self.frame.set(None);
    }

    /// What the render target made of the published frame.
    ///
    /// [`ImageStatus::Pending`] while no frame is published and until a
    /// target first draws the published frame, then [`ImageStatus::Ready`] —
    /// the frame is on screen — or [`ImageStatus::Refused`] with the reason
    /// the target rejected the frame's pixels: they exceed its image limits,
    /// or its image budget cannot hold them beside the images already
    /// registered. A refused frame draws nothing. Publishing a different
    /// frame follows that frame's own status, and clearing returns to
    /// `Pending`.
    #[must_use]
    pub fn status(&self) -> Computed<ImageStatus> {
        flatten_signal(self.frame.clone().map(|frame| {
            frame.map_or_else(
                || Computed::constant(ImageStatus::Pending),
                |frame| frame.status(),
            )
        }))
    }

    /// The displayed frame's pixel dimensions, which the view's layout reads
    /// so a size change re-lays out without a rebuild.
    fn dimensions(&self) -> Computed<Option<(u32, u32)>> {
        self.frame
            .clone()
            .map(|frame| frame.map(|frame| (frame.width(), frame.height())))
            .computed()
    }
}

/// An image view whose decoded frame can change without rebuilding its
/// subtree.
///
/// One [`Canvas`] mounts for the view's whole life and re-records only the
/// image command when a frame is published, so nothing remounts and no state
/// inside the subtree is lost.
#[derive(Debug)]
pub struct ReactiveImage {
    handle: ReactiveImageHandle,
    resizable: bool,
    content_mode: Option<ContentMode>,
}

impl ReactiveImage {
    /// Allows this image to stretch to its proposed bounds.
    ///
    /// Otherwise it keeps its natural size and only scales down, preserving its
    /// aspect ratio and claiming no leftover space. See [`Image::resizable`].
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
        let dimensions = self.handle.dimensions();
        let natural = dimensions
            .map(|dimensions| dimensions.and_then(|(width, height)| pixel_size(width, height)));
        let mode = self.content_mode;
        let frame = self.handle.frame;
        // Publishing a frame records the canvas' one image command again, in
        // place; nothing else re-records.
        let canvas = Canvas::new(move |cx| {
            cx.scope(frame, move |cx, frame: &Option<Frame>| {
                if let Some(frame) = frame {
                    frame.draw(cx, mode);
                }
            });
        })
        .intrinsic_size(natural);
        sized(canvas, self.resizable, dimensions)
    }
}

/// Creates one persistent image view and its precise frame-update handle.
#[must_use]
pub fn reactive_image() -> (ReactiveImageHandle, ReactiveImage) {
    let handle = ReactiveImageHandle {
        frame: Binding::container(None),
    };
    (
        handle.clone(),
        ReactiveImage {
            handle,
            resizable: false,
            content_mode: None,
        },
    )
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
    use super::{Image, Interpolation, reactive_image, rgba16f_to_srgb8};
    use half::f16;
    use waterui_core::Signal;
    use waterui_graphics::canvas::{ImageStatus, Sampling};

    #[test]
    fn reactive_image_publishes_the_latest_frame() {
        let (handle, _view) = reactive_image();
        let dimensions = handle.dimensions();
        let status = handle.status();
        assert_eq!(dimensions.snapshot(), None);
        assert_eq!(status.snapshot(), ImageStatus::Pending);

        handle.set(Image::new(vec![0, 0, 0, 255], 1, 1));
        assert_eq!(dimensions.snapshot(), Some((1, 1)));
        assert_eq!(status.snapshot(), ImageStatus::Pending);

        handle.set(Image::new(vec![0; 2 * 3 * 4], 2, 3).interpolation(Interpolation::Nearest));
        assert_eq!(dimensions.snapshot(), Some((2, 3)));
        let frame = handle
            .frame
            .snapshot()
            .expect("the published frame is on display");
        assert_eq!(frame.sampling, Sampling::Nearest);

        handle.clear();
        assert_eq!(dimensions.snapshot(), None);
        assert_eq!(status.snapshot(), ImageStatus::Pending);
    }

    #[test]
    fn interpolation_selects_the_sampling() {
        let image = Image::new(alloc::vec![0, 0, 0, 255], 1, 1);
        assert_eq!(image.frame.sampling, Sampling::Linear);
        assert_eq!(
            image.interpolation(Interpolation::Nearest).frame.sampling,
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
