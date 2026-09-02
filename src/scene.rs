//! How a decoded image becomes scene commands.
//!
//! An image is one [`Scene2D::draw_image`] call. `draw_image` paints the
//! brush's own pixel rectangle — one image pixel to one unit — so everything
//! the component decides about placement lives in the transform handed to it:
//! the scale that resizes the pixel grid onto the box layout gave the view, and
//! the translation that centres it when the two aspect ratios disagree.
//!
//! Nothing here knows which engine is listening. The same commands drive the
//! GPU compute rasterizer, the CPU sparse-strip rasterizer, and a backend that
//! merges them into a scene of its own.

use kurbo::{Affine, Point, Rect, Shape as _, Size};
use peniko::{Fill, ImageBrush};
use waterui_graphics::{Scene2D, SceneContent, SceneInvalidator};
use waterui_layout::ContentMode;

/// Where an image's pixel grid lands inside the box the layout gave the view.
///
/// `mode` is [`None`] for the unconstrained case — the image stretches to the
/// box on both axes independently, which is what a plain `.resizable()` asks
/// for. [`ContentMode::Fit`] and [`ContentMode::Fill`] scale both axes by one
/// factor instead and centre the result, leaving slack or overflowing.
fn destination(image: Size, bounds: Size, mode: Option<ContentMode>) -> Rect {
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
fn overflows(destination: Rect, bounds: Size) -> bool {
    destination.x0 < 0.0
        || destination.y0 < 0.0
        || destination.x1 > bounds.width
        || destination.y1 > bounds.height
}

/// Draws `brush` across a `width` x `height` box, placed according to `mode`.
///
/// Draws nothing when either the image or the box has no area: there is no
/// meaningful scale between them, and the transform would be degenerate.
pub fn draw(
    scene: &mut dyn Scene2D,
    brush: &ImageBrush,
    mode: Option<ContentMode>,
    width: f32,
    height: f32,
) {
    let image = Size::new(f64::from(brush.image.width), f64::from(brush.image.height));
    let bounds = Size::new(f64::from(width), f64::from(height));
    if image.width <= 0.0 || image.height <= 0.0 || bounds.width <= 0.0 || bounds.height <= 0.0 {
        return;
    }

    let destination = destination(image, bounds, mode);
    let transform = Affine::translate((destination.x0, destination.y0))
        * Affine::scale_non_uniform(
            destination.width() / image.width,
            destination.height() / image.height,
        );

    let clipped = overflows(destination, bounds);
    if clipped {
        scene.push_clip_layer(
            Fill::NonZero,
            Affine::IDENTITY,
            &Rect::from_origin_size(Point::ZERO, bounds).to_path(0.0),
        );
    }
    scene.draw_image(brush, transform);
    if clipped {
        scene.pop_layer();
    }
}

/// Scene content that draws one decoded image for the lifetime of a view.
pub struct ImageSceneContent {
    brush: ImageBrush,
    mode: Option<ContentMode>,
}

impl ImageSceneContent {
    pub const fn new(brush: ImageBrush, mode: Option<ContentMode>) -> Self {
        Self { brush, mode }
    }
}

impl core::fmt::Debug for ImageSceneContent {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ImageSceneContent")
            .field("width", &self.brush.image.width)
            .field("height", &self.brush.image.height)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl SceneContent for ImageSceneContent {
    fn build_scene(&mut self, scene: &mut dyn Scene2D, width: f32, height: f32) -> bool {
        draw(scene, &self.brush, self.mode, width, height);
        false
    }

    fn set_invalidator(&mut self, _invalidator: Option<SceneInvalidator>) {}
}

#[cfg(test)]
mod tests {
    use super::{destination, draw, overflows};
    use alloc::vec::Vec;
    use kurbo::{Affine, Point, Rect, Size};
    use peniko::{Blob, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageSampler};
    use waterui_graphics::{GlyphRun, Scene2D, SceneRecording};
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

    /// Records which commands a scene was given, and with what transform.
    #[derive(Default)]
    struct Commands {
        images: Vec<Affine>,
        clips: usize,
        pops: usize,
    }

    impl Scene2D for Commands {
        fn fill(
            &mut self,
            _fill: peniko::Fill,
            _transform: Affine,
            _brush: &peniko::Brush,
            _brush_transform: Option<Affine>,
            _shape: &kurbo::BezPath,
        ) {
        }

        fn stroke(
            &mut self,
            _stroke: &kurbo::Stroke,
            _transform: Affine,
            _brush: &peniko::Brush,
            _brush_transform: Option<Affine>,
            _shape: &kurbo::BezPath,
        ) {
        }

        fn push_layer(
            &mut self,
            _fill: peniko::Fill,
            _blend: peniko::BlendMode,
            _alpha: f32,
            _transform: Affine,
            _clip: &kurbo::BezPath,
        ) {
            self.clips += 1;
        }

        fn push_clip_layer(
            &mut self,
            _fill: peniko::Fill,
            _transform: Affine,
            _clip: &kurbo::BezPath,
        ) {
            self.clips += 1;
        }

        fn pop_layer(&mut self) {
            self.pops += 1;
        }

        fn draw_image(&mut self, _image: &ImageBrush, transform: Affine) {
            self.images.push(transform);
        }

        fn draw_glyph_run(&mut self, _run: &GlyphRun<'_>) {}

        fn reset(&mut self) {
            self.images.clear();
            self.clips = 0;
            self.pops = 0;
        }
    }

    fn brush(width: u32, height: u32) -> ImageBrush {
        let pixels = alloc::vec![255_u8; (width as usize) * (height as usize) * 4];
        ImageBrush {
            image: ImageData {
                data: Blob::from(pixels),
                format: ImageFormat::Rgba8,
                alpha_type: ImageAlphaType::Alpha,
                width,
                height,
            },
            sampler: ImageSampler::default(),
        }
    }

    #[test]
    fn stretch_maps_the_pixel_grid_onto_the_whole_box() {
        let mut commands = Commands::default();
        draw(&mut commands, &brush(80, 20), None, 100.0, 100.0);

        assert_eq!(commands.clips, 0);
        assert_eq!(commands.pops, 0);
        let [transform] = commands.images[..] else {
            panic!("stretching must draw exactly one image");
        };
        // The image's own corners land on the box's corners.
        assert_eq!(transform * Point::ZERO, Point::ZERO);
        assert_eq!(transform * Point::new(80.0, 20.0), Point::new(100.0, 100.0));
    }

    #[test]
    fn fill_clips_the_overflow_and_fit_does_not() {
        let mut filled = Commands::default();
        draw(
            &mut filled,
            &brush(80, 20),
            Some(ContentMode::Fill),
            100.0,
            100.0,
        );
        assert_eq!((filled.clips, filled.pops, filled.images.len()), (1, 1, 1));
        // 5x scale, centred: the left edge starts 150 points off the box.
        assert_eq!(filled.images[0] * Point::ZERO, Point::new(-150.0, 0.0));

        let mut fitted = Commands::default();
        draw(
            &mut fitted,
            &brush(80, 20),
            Some(ContentMode::Fit),
            100.0,
            100.0,
        );
        assert_eq!((fitted.clips, fitted.pops, fitted.images.len()), (0, 0, 1));
        assert_eq!(fitted.images[0] * Point::ZERO, Point::new(0.0, 37.5));
    }

    #[test]
    fn a_degenerate_box_or_image_draws_nothing() {
        let mut commands = Commands::default();
        draw(&mut commands, &brush(80, 20), None, 0.0, 100.0);
        draw(&mut commands, &brush(0, 0), None, 100.0, 100.0);
        assert!(commands.images.is_empty());
    }

    /// The recording is what a backend replays, so the commands have to survive
    /// being recorded rather than only reaching a live scene.
    #[test]
    fn commands_survive_a_recording() {
        let mut recording = SceneRecording::new();
        draw(
            &mut recording,
            &brush(80, 20),
            Some(ContentMode::Fill),
            100.0,
            100.0,
        );
        assert_eq!(recording.len(), 3, "clip, image, pop");

        let mut replayed = Commands::default();
        recording.replay(&mut replayed, Some(Affine::translate((10.0, 0.0))));
        assert_eq!(
            (replayed.clips, replayed.pops, replayed.images.len()),
            (1, 1, 1)
        );
        assert_eq!(replayed.images[0] * Point::ZERO, Point::new(-140.0, 0.0));
    }
}
