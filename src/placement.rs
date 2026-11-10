//! Where a decoded image lands on its canvas.
//!
//! An image is one `image` command. Its destination rectangle is computed from
//! the canvas size by the content mode, so a box the layout resizes moves the
//! one retained command in place; nothing records again.
//!
//! The render target registers the pixels itself, on first use, and may refuse
//! them — an image beyond the device's limits, or one its image budget cannot
//! hold beside the others. A refused image draws nothing, and the refusal is
//! the image's [`status`](ImageSource::status).

use alloc::vec::Vec;

use num_traits::ToPrimitive;
use waterui_core::layout::{Point, Rect, Size};
use waterui_core::{Computed, SignalExt as _};
#[cfg(feature = "gpu")]
use waterui_graphics::canvas::DrawingCx;
use waterui_graphics::canvas::{Cx, ImageSource, ImageStatus, PixelBuffer, PixelFormat, Sampling};
use waterui_layout::ContentMode;

/// One decoded picture and how it is sampled: what a canvas draws as an image.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    source: ImageSource,
    pub(crate) sampling: Sampling,
}

impl Frame {
    /// Straight-alpha sRGB8 `pixels` of a `width` x `height` grid.
    ///
    /// # Panics
    /// When the grid has no area, or `pixels` is not exactly
    /// `width * height * 4` bytes.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, sampling: Sampling) -> Self {
        let source = ImageSource::new(PixelBuffer::new(width, height, PixelFormat::Rgba8, pixels))
            .expect("an image needs pixels: a non-zero size and width * height * 4 bytes");
        Self { source, sampling }
    }

    /// Width of the pixel grid.
    pub fn width(&self) -> u32 {
        self.source.width()
    }

    /// Height of the pixel grid.
    pub fn height(&self) -> u32 {
        self.source.height()
    }

    /// This frame, sampled with `sampling` instead.
    pub const fn sampled(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// How the render target's registration of the pixels stands.
    pub fn status(&self) -> Computed<ImageStatus> {
        self.source.status()
    }

    /// The size the pixel grid *is*, at one pixel per unit.
    pub fn size(&self) -> Option<Size> {
        pixel_size(self.width(), self.height())
    }

    /// Draws the frame placed by `mode` in the box of `cx`, which layout may
    /// resize.
    pub fn draw(&self, cx: &mut Cx, mode: Option<ContentMode>) {
        let size = cx.size();
        let (source, sampling) = (self.source.clone(), self.sampling);
        let image = self.pixels();
        let destination = size.map(move |bounds| destination(image, bounds, mode));
        if clips(mode) {
            cx.clip(size.map(Rect::from_size), |cx| {
                cx.image(source, destination, sampling);
            });
        } else {
            cx.image(source, destination, sampling);
        }
    }

    /// Draws the frame placed by `mode` in a recording of `bounds`.
    #[cfg(feature = "gpu")]
    pub fn record(&self, cx: &mut DrawingCx, bounds: Size, mode: Option<ContentMode>) {
        let destination = destination(self.pixels(), bounds, mode);
        if clips(mode) {
            cx.clip(Rect::from_size(bounds), |cx| {
                cx.image(&self.source, destination, self.sampling);
            });
        } else {
            cx.image(&self.source, destination, self.sampling);
        }
    }

    fn pixels(&self) -> Size {
        Size::new(u32_to_f32(self.width()), u32_to_f32(self.height()))
    }
}

/// `value` as an `f32`, which every pixel dimension and box side fits.
pub fn u32_to_f32(value: u32) -> f32 {
    value
        .to_f32()
        .expect("image dimensions must be representable as f32")
}

/// The size a `width` x `height` pixel grid *is*, at one pixel per unit.
///
/// This is what the canvas' intrinsic size answers for an image: layout falls
/// back to it where nothing else settles the question, and derives an open axis
/// from it when a container names only the other one. `None` when either axis
/// is zero — an image with no pixels has no size.
pub fn pixel_size(width: u32, height: u32) -> Option<Size> {
    (width > 0 && height > 0).then(|| Size::new(u32_to_f32(width), u32_to_f32(height)))
}

/// Whether `mode` can leave the box, and so needs clipping to stay inside it.
///
/// Only [`ContentMode::Fill`] ever does. A surface of its own would clip at its
/// edges anyway, but a canvas merged into a parent's drawing would not, so the
/// clip is part of the drawing rather than a property of the target.
const fn clips(mode: Option<ContentMode>) -> bool {
    matches!(mode, Some(ContentMode::Fill))
}

/// Where an image of `image` pixels lands inside the box `bounds`.
///
/// `mode` is [`None`] for the unconstrained case — the image stretches to the
/// box on both axes independently, which is what a plain `.resizable()` asks
/// for. [`ContentMode::Fit`] and [`ContentMode::Fill`] scale both axes by one
/// factor instead and centre the result, leaving slack or overflowing. A box
/// with no area has an empty destination.
fn destination(image: Size, bounds: Size, mode: Option<ContentMode>) -> Rect {
    if bounds.width <= 0.0 || bounds.height <= 0.0 {
        return Rect::from_size(Size::zero());
    }
    let Some(mode) = mode else {
        return Rect::from_size(bounds);
    };
    let horizontal = bounds.width / image.width;
    let vertical = bounds.height / image.height;
    let scale = match mode {
        ContentMode::Fit => horizontal.min(vertical),
        ContentMode::Fill => horizontal.max(vertical),
    };
    let size = Size::new(image.width * scale, image.height * scale);
    let origin = Point::new(
        (bounds.width - size.width) / 2.0,
        (bounds.height - size.height) / 2.0,
    );
    Rect::new(origin, size)
}

#[cfg(test)]
mod tests {
    use super::{Frame, Sampling, Size, clips, destination, pixel_size};
    use waterui_core::layout::Rect;
    use waterui_layout::ContentMode;

    /// A 4:1 image in a square box: the two aspect ratios disagree, so every
    /// mode resolves to a different rectangle.
    const WIDE: Size = Size::new(80.0, 20.0);
    const SQUARE: Size = Size::new(100.0, 100.0);

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect::new(
            waterui_core::layout::Point::new(x0, y0),
            Size::new(x1 - x0, y1 - y0),
        )
    }

    #[test]
    fn no_mode_stretches_to_the_whole_box() {
        assert_eq!(
            destination(WIDE, SQUARE, None),
            rect(0.0, 0.0, 100.0, 100.0)
        );
    }

    #[test]
    fn fit_keeps_the_aspect_ratio_inside_the_box() {
        // 100/80 = 1.25 horizontally against 100/20 = 5 vertically; fit takes
        // the smaller, so the image spans the full width and is centred in the
        // 75 points of slack left on the vertical axis.
        assert_eq!(
            destination(WIDE, SQUARE, Some(ContentMode::Fit)),
            rect(0.0, 37.5, 100.0, 62.5)
        );
    }

    #[test]
    fn fill_covers_the_box_and_only_fill_clips() {
        // Fill takes the larger factor, 5, so the image becomes 400x100 and
        // hangs 150 points off each side of the square.
        assert_eq!(
            destination(WIDE, SQUARE, Some(ContentMode::Fill)),
            rect(-150.0, 0.0, 250.0, 100.0)
        );
        assert!(clips(Some(ContentMode::Fill)));
        assert!(!clips(Some(ContentMode::Fit)));
        assert!(!clips(None));
    }

    #[test]
    fn a_degenerate_box_has_an_empty_destination() {
        for mode in [None, Some(ContentMode::Fit), Some(ContentMode::Fill)] {
            let empty = destination(WIDE, Size::new(0.0, 100.0), mode);
            assert_eq!((empty.width(), empty.height()), (0.0, 0.0));
        }
    }

    /// The natural size is the pixel grid at one pixel per unit, whatever the
    /// content mode: the mode decides where the pixels land inside a box, not
    /// how big the box wants to be.
    #[test]
    fn an_image_is_its_pixel_grid() {
        let frame = Frame::new(vec![255; 80 * 20 * 4], 80, 20, Sampling::Linear);
        assert_eq!(frame.size(), Some(WIDE));
        assert_eq!(pixel_size(80, 0), None);
        assert_eq!(pixel_size(0, 0), None);
    }
}
