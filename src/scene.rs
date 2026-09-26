//! How decoded pixels become Cherenkov content.
//!
//! An image is one [`Draw::image`] call: the engine that will draw it uploads
//! the pixels once and hands back a handle, and the recording names that handle
//! with the destination rectangle the content mode resolves — the box the
//! layout gave the view, or the fitted or filling rectangle centred in it.
//!
//! Nothing here knows which backend is listening: the same content renders on
//! `cherenkov-gpu`, on the CPU backend, and in any backend that merges it into
//! a scene of its own.

use alloc::rc::Rc;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use cherenkov::kurbo::{Point, Rect, Size};
use cherenkov::{Draw as _, Image as ImageHandle, ImageData, ResourceError, Rgba8, Sampling};
use num_traits::ToPrimitive;
use waterui_core::layout::Size as LayoutSize;
use waterui_graphics::{Scene, SceneContent, SceneInvalidator, SceneResources};
use waterui_layout::ContentMode;

pub fn u32_to_f32(value: u32) -> f32 {
    value
        .to_f32()
        .expect("image dimensions must be representable as f32")
}

/// Decoded straight-alpha sRGB8 pixels, shared between the views that show
/// them and the engine that uploads them.
#[derive(Clone, PartialEq, Eq)]
pub struct Pixels {
    /// Columns in the grid.
    pub width: u32,
    /// Rows in the grid.
    pub height: u32,
    /// `width * height * 4` bytes, row-major.
    pub data: Arc<[u8]>,
}

impl fmt::Debug for Pixels {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Pixels")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl Pixels {
    /// `width * height * 4` bytes of RGBA8.
    ///
    /// # Panics
    /// When the byte count does not match the dimensions.
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "Pixel data length must be width * height * 4"
        );
        Self {
            width,
            height,
            data: Arc::from(data),
        }
    }

    /// The pixel grid as an upload; `None` when either axis is zero, which the
    /// engine rejects and which draws nothing anyway.
    fn upload(&self) -> Option<ImageData<Rgba8>> {
        ImageData::<Rgba8>::new(self.width, self.height, Arc::clone(&self.data)).ok()
    }

    /// Whether two `Pixels` share the same allocation.
    fn same_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

/// The size a `width` x `height` pixel grid *is*, at one pixel per unit.
///
/// This is what [`SceneContent::intrinsic_size`] answers for an image: layout
/// falls back to it where nothing else settles the question, and derives an
/// open axis from it when a container names only the other one. `None` when
/// either axis is zero — an image with no pixels has no size, and the hook
/// wants an honest `None` rather than a degenerate one.
pub fn pixel_size(width: u32, height: u32) -> Option<LayoutSize> {
    (width > 0 && height > 0).then(|| LayoutSize::new(u32_to_f32(width), u32_to_f32(height)))
}

/// Where an image's pixel grid lands inside the box the layout gave the view.
///
/// `mode` is [`None`] for the unconstrained case — the image stretches to the
/// box on both axes independently, which is what a plain `.resizable()` asks
/// for. [`ContentMode::Fit`] and [`ContentMode::Fill`] scale both axes by one
/// factor instead and centre the result, leaving slack or overflowing.
pub fn destination(image: Size, bounds: Size, mode: Option<ContentMode>) -> Rect {
    let Some(mode) = mode else {
        return Rect::from_origin_size(Point::ZERO, bounds);
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
    Rect::from_origin_size(origin, size)
}

/// Whether `destination` leaves the box, and so needs clipping to stay inside it.
///
/// Only [`ContentMode::Fill`] ever does. A surface of its own would clip at its
/// edges anyway, but content merged into a parent scene would not, so the clip
/// is part of the drawing rather than a property of the target.
pub fn overflows(destination: Rect, bounds: Size) -> bool {
    destination.x0 < 0.0
        || destination.y0 < 0.0
        || destination.x1 > bounds.width
        || destination.y1 > bounds.height
}

/// The rectangle a `pixels`-sized image is drawn into across a
/// `width` x `height` box, and whether that rectangle needs the box as a clip.
///
/// `None` when either the image or the box has no area: there is no meaningful
/// scale between them.
pub fn placement(
    pixels: (u32, u32),
    mode: Option<ContentMode>,
    width: f32,
    height: f32,
) -> Option<(Rect, bool)> {
    let image = Size::new(f64::from(pixels.0), f64::from(pixels.1));
    let bounds = Size::new(f64::from(width), f64::from(height));
    if image.width <= 0.0 || image.height <= 0.0 || bounds.width <= 0.0 || bounds.height <= 0.0 {
        return None;
    }
    let destination = destination(image, bounds, mode);
    Some((destination, overflows(destination, bounds)))
}

/// One image's engine handle, minted on the first record against an engine
/// and reused until the pixels change or the engine does.
#[derive(Default)]
pub struct Uploaded {
    pixels: Option<Pixels>,
    resources: Option<Rc<dyn SceneResources>>,
    handle: Option<ImageHandle<Rgba8>>,
}

impl Uploaded {
    fn handle(
        &mut self,
        pixels: &Pixels,
        resources: &Rc<dyn SceneResources>,
    ) -> Result<Option<&ImageHandle<Rgba8>>, ResourceError> {
        let current = self
            .pixels
            .as_ref()
            .is_some_and(|uploaded| uploaded.same_allocation(pixels))
            && self
                .resources
                .as_ref()
                .is_some_and(|engine| Rc::ptr_eq(engine, resources));
        if !current {
            self.handle = match pixels.upload() {
                Some(upload) => Some(resources.image(upload)?),
                None => None,
            };
            self.pixels = Some(pixels.clone());
            self.resources = Some(Rc::clone(resources));
        }
        Ok(self.handle.as_ref())
    }
}

/// Records `pixels` across the scene's box, placed according to `mode`.
///
/// Draws nothing when the image or the box has no area.
///
/// # Errors
/// [`ResourceError`] when the engine refuses the upload.
pub fn draw(
    scene: &mut Scene<'_>,
    uploaded: &mut Uploaded,
    pixels: &Pixels,
    sampling: Sampling,
    mode: Option<ContentMode>,
) -> Result<(), ResourceError> {
    let Some((destination, clipped)) = placement(
        (pixels.width, pixels.height),
        mode,
        scene.width(),
        scene.height(),
    ) else {
        return Ok(());
    };
    let Some(handle) = uploaded.handle(pixels, scene.resources())? else {
        return Ok(());
    };
    let id = handle.id();
    let bounds = Rect::new(
        0.0,
        0.0,
        f64::from(scene.width()),
        f64::from(scene.height()),
    );
    let recorder = scene.recorder();
    if clipped {
        recorder.clip(bounds, |recorder| {
            recorder.image(id, destination, sampling);
        });
    } else {
        recorder.image(id, destination, sampling);
    }
    Ok(())
}

/// Scene content that draws one decoded image for the lifetime of a view.
pub struct ImageSceneContent {
    pixels: Pixels,
    sampling: Sampling,
    mode: Option<ContentMode>,
    uploaded: Uploaded,
}

impl ImageSceneContent {
    /// Content drawing `pixels` with `sampling`, placed by `mode`.
    #[must_use]
    pub fn new(pixels: Pixels, sampling: Sampling, mode: Option<ContentMode>) -> Self {
        Self {
            pixels,
            sampling,
            mode,
            uploaded: Uploaded::default(),
        }
    }
}

impl fmt::Debug for ImageSceneContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImageSceneContent")
            .field("width", &self.pixels.width)
            .field("height", &self.pixels.height)
            .field("sampling", &self.sampling)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl SceneContent for ImageSceneContent {
    fn record(&mut self, scene: &mut Scene<'_>) -> bool {
        if let Err(error) = draw(
            scene,
            &mut self.uploaded,
            &self.pixels,
            self.sampling,
            self.mode,
        ) {
            tracing::error!(%error, "image upload rejected by the engine");
        }
        false
    }

    fn intrinsic_size(&self) -> Option<LayoutSize> {
        pixel_size(self.pixels.width, self.pixels.height)
    }

    fn set_invalidator(&mut self, _invalidator: Option<SceneInvalidator>) {}
}

/// The pixels a [`ReactiveImage`](crate::ReactiveImage) currently shows, and
/// the hook that re-records its content when they change.
pub struct ReactiveImageState {
    pub pixels: RefCell<Option<(Pixels, Sampling)>>,
    pub dimensions: waterui_core::Binding<Option<(u32, u32)>>,
    pub invalidator: RefCell<Option<SceneInvalidator>>,
}

impl fmt::Debug for ReactiveImageState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReactiveImageState")
            .field(
                "dimensions",
                &waterui_core::Signal::snapshot(&self.dimensions),
            )
            .finish_non_exhaustive()
    }
}

/// Scene content that draws whatever pixels its shared state currently holds.
pub struct ReactiveImageSceneContent {
    pub state: Rc<ReactiveImageState>,
    pub content_mode: Option<ContentMode>,
    uploaded: Uploaded,
}

impl ReactiveImageSceneContent {
    pub fn new(state: Rc<ReactiveImageState>, content_mode: Option<ContentMode>) -> Self {
        Self {
            state,
            content_mode,
            uploaded: Uploaded::default(),
        }
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
    fn record(&mut self, scene: &mut Scene<'_>) -> bool {
        let shown = self.state.pixels.borrow();
        if let Some((pixels, sampling)) = shown.as_ref()
            && let Err(error) = draw(
                scene,
                &mut self.uploaded,
                pixels,
                *sampling,
                self.content_mode,
            )
        {
            tracing::error!(%error, "image upload rejected by the engine");
        }
        false
    }

    fn intrinsic_size(&self) -> Option<LayoutSize> {
        waterui_core::Signal::snapshot(&self.state.dimensions)
            .and_then(|(width, height)| pixel_size(width, height))
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

#[cfg(test)]
mod tests {
    use super::{ImageSceneContent, LayoutSize, Pixels, destination, overflows, placement};
    use cherenkov::Sampling;
    use cherenkov::kurbo::{Point, Rect, Size};
    use waterui_graphics::SceneContent as _;
    use waterui_layout::ContentMode;

    /// A 4:1 image in a square box: the two aspect ratios disagree, so every
    /// mode resolves to a different rectangle.
    const WIDE: Size = Size::new(80.0, 20.0);
    const SQUARE: Size = Size::new(100.0, 100.0);

    #[test]
    fn no_mode_stretches_to_the_whole_box() {
        assert_eq!(
            destination(WIDE, SQUARE, None),
            Rect::new(0.0, 0.0, 100.0, 100.0)
        );
    }

    #[test]
    fn fit_keeps_the_aspect_ratio_inside_the_box() {
        // 100/80 = 1.25 horizontally against 100/20 = 5 vertically; fit takes
        // the smaller, so the image spans the full width and is centred in the
        // 75 points of slack left on the vertical axis.
        assert_eq!(
            destination(WIDE, SQUARE, Some(ContentMode::Fit)),
            Rect::new(0.0, 37.5, 100.0, 62.5)
        );
    }

    #[test]
    fn fill_covers_the_box_and_overflows_the_long_axis() {
        // Fill takes the larger factor, 5, so the image becomes 400x100 and
        // hangs 150 points off each side of the square.
        let rect = destination(WIDE, SQUARE, Some(ContentMode::Fill));
        assert_eq!(rect, Rect::new(-150.0, 0.0, 250.0, 100.0));
        assert!(overflows(rect, SQUARE));
        assert!(!overflows(
            destination(WIDE, SQUARE, Some(ContentMode::Fit)),
            SQUARE
        ));
        assert!(!overflows(destination(WIDE, SQUARE, None), SQUARE));
    }

    fn pixels(width: u32, height: u32) -> Pixels {
        Pixels::new(
            alloc::vec![255_u8; (width as usize) * (height as usize) * 4],
            width,
            height,
        )
    }

    #[test]
    fn stretch_maps_the_pixel_grid_onto_the_whole_box() {
        let (rect, clipped) = placement((80, 20), None, 100.0, 100.0).expect("has area");
        assert!(!clipped);
        assert_eq!(rect.origin(), Point::ZERO);
        assert_eq!(rect.size(), Size::new(100.0, 100.0));
    }

    #[test]
    fn fill_clips_the_overflow_and_fit_does_not() {
        let (filled, clipped) =
            placement((80, 20), Some(ContentMode::Fill), 100.0, 100.0).expect("has area");
        assert!(clipped);
        // 5x scale, centred: the left edge starts 150 points off the box.
        assert_eq!(filled.origin(), Point::new(-150.0, 0.0));

        let (fitted, clipped) =
            placement((80, 20), Some(ContentMode::Fit), 100.0, 100.0).expect("has area");
        assert!(!clipped);
        assert_eq!(fitted.origin(), Point::new(0.0, 37.5));
    }

    #[test]
    fn a_degenerate_box_or_image_draws_nothing() {
        assert!(placement((80, 20), None, 0.0, 100.0).is_none());
        assert!(placement((0, 0), None, 100.0, 100.0).is_none());
    }

    /// The natural size is the pixel grid at one pixel per unit, whatever the
    /// content mode: the mode decides where the pixels land inside a box, not
    /// how big the box wants to be.
    #[test]
    fn an_image_is_its_pixel_grid() {
        for mode in [None, Some(ContentMode::Fit), Some(ContentMode::Fill)] {
            let content = ImageSceneContent::new(pixels(80, 20), Sampling::Linear, mode);
            assert_eq!(content.intrinsic_size(), Some(LayoutSize::new(80.0, 20.0)));
        }
    }

    /// An image with no pixels has no size of its own.
    #[test]
    fn an_empty_image_has_no_size() {
        assert_eq!(
            ImageSceneContent::new(pixels(0, 0), Sampling::Linear, None).intrinsic_size(),
            None
        );
        assert_eq!(
            ImageSceneContent::new(pixels(80, 0), Sampling::Linear, None).intrinsic_size(),
            None
        );
    }
}
