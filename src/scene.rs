//! How a decoded image becomes engine content.
//!
//! An image is one `Draw::image` call. [`SceneContent::build_scene`]
//! registers the pixel grid with the recording's [`RecordingResources`] on the frame
//! that first draws it and names the registration's [`ImageId`] against
//! the destination rectangle the content mode resolves — the box the layout
//! gave the view, or the fitted or filling rectangle centred in it. The
//! [`Registered`] handle stays with the content for as long as its
//! recordings name the image.
//!
//! The engine may refuse a registration — an image beyond its
//! [`ImageLimits`](waterui_graphics::draw::ImageLimits), or one its image
//! budget cannot take beside the images already registered. A refused image
//! draws nothing, is not offered to the same engine again, and is reported:
//! through `tracing`, and as [`RegistrationState::Refused`] on the
//! registration state of the view that owns it.
//!
//! Nothing here knows which backend is listening: the same commands render on
//! `cherenkov-gpu`, on the CPU rasterizer, and in a backend that merges them
//! into a scene of its own.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use num_traits::ToPrimitive;
use waterui_core::layout::Size as LayoutSize;
use waterui_core::{Binding, Signal as _};
use waterui_graphics::draw::kurbo::{Point, Rect, Size};
use waterui_graphics::draw::{Draw as _, ImageId, Recorder, Sampling};
use waterui_graphics::{
    ImageData, RecordingResources, Registered, ResourceError, Rgba8, SceneContent,
};
use waterui_layout::ContentMode;

/// The scene backend refused to register an image's pixels.
///
/// An image whose registration is refused draws nothing. The usual causes are
/// an image larger than the backend's image limits and one its image budget
/// cannot hold beside the images already registered; [`Self::resource_error`]
/// says which.
///
/// Cheap to clone: every clone shares the one backend error.
#[derive(Debug, Clone)]
pub struct RegistrationError(Arc<ResourceError>);

impl RegistrationError {
    /// The error the scene backend refused the registration with.
    #[must_use]
    pub fn resource_error(&self) -> &ResourceError {
        &self.0
    }
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "the scene backend rejected the image: {}",
            self.0
        )
    }
}

impl core::error::Error for RegistrationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// What registering one pixel grid with an engine came to: the live
/// registration to draw, or the refusal that leaves the image blank.
pub type Registration = Result<Registered<ImageId>, RegistrationError>;

/// How the registration of the frame an image view displays stands.
///
/// A view starts at [`Pending`](Self::Pending); each recording pass that
/// registers or refuses the published frame settles it, and publishing a
/// different frame — or clearing it — returns it to `Pending`. The state
/// changes from the main-thread executor after the recording pass that
/// settled it, never from inside rendering; see [`report`].
#[derive(Debug, Clone)]
pub enum RegistrationState {
    /// No recording pass has settled the current frame's registration yet.
    Pending,
    /// The scene backend registered the current frame's pixels; the image
    /// is on screen.
    Registered,
    /// The scene backend refused the current frame's pixels — they exceed
    /// its image limits, or its image budget cannot hold them beside the
    /// images already registered. The image draws nothing.
    Refused(RegistrationError),
}

/// Registers `data` with the recording's engine, reporting a refusal
/// through `tracing`.
pub fn register(data: ImageData<Rgba8>, resources: &RecordingResources<'_>) -> Registration {
    let (width, height) = (data.width(), data.height());
    resources.image(data).map_err(|error| {
        tracing::error!(
            width,
            height,
            %error,
            "the scene backend rejected an image registration; the image draws nothing"
        );
        RegistrationError(Arc::new(error))
    })
}

/// Brings a view's registration state in line with `outcome` —
/// [`Refused`](RegistrationState::Refused) on a refusal of the pixels it
/// displays, [`Registered`](RegistrationState::Registered) when they
/// registered — once the recording pass that settled it is over.
///
/// `build_scene` runs inside the host's recording pass, with the content
/// borrowed; a watcher of the registration state is application code, so the
/// change is made from the main-thread executor instead of inside that
/// borrow. `current` is asked again when the task runs, and an outcome the
/// view has moved past by then — a different frame was published meanwhile —
/// is dropped. Nothing is sent while the state already holds the outcome's
/// variant.
pub fn report(
    registration: &Binding<RegistrationState>,
    outcome: Option<&RegistrationError>,
    current: impl FnOnce() -> bool + 'static,
) {
    let settled = outcome.map_or(RegistrationState::Registered, |error| {
        RegistrationState::Refused(error.clone())
    });
    if core::mem::discriminant(&registration.snapshot()) == core::mem::discriminant(&settled) {
        return;
    }
    let registration = registration.clone();
    executor_core::spawn_local(async move {
        if current() {
            registration.set(settled);
        }
    })
    .detach();
}

/// Decoded straight-alpha sRGB8 pixels, shared between the view that owns them
/// and the mounts that upload them.
///
/// `Pixels` is cheap to clone — the view, every mounted
/// [`ImageSceneContent`], and the engine registration it becomes share the one
/// allocation.
#[derive(Clone, PartialEq, Eq)]
pub struct Pixels {
    /// Width of the grid in pixels.
    pub width: u32,
    /// Height of the grid in pixels.
    pub height: u32,
    /// The pixel data, `width * height * 4` bytes in straight-alpha sRGB8.
    pub data: Arc<[u8]>,
}

impl Pixels {
    /// Wraps `data` as a `width` x `height` grid.
    ///
    /// # Panics
    /// When `data` is not exactly `width * height * 4` bytes.
    #[must_use]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "pixel data must be width * height * 4 bytes"
        );
        Self {
            width,
            height,
            data: Arc::from(data),
        }
    }

    /// Packages the grid for engine upload. `None` when there is nothing to
    /// upload — the engine rejects a zero-area grid, and a mount with no
    /// pixels draws nothing.
    pub(crate) fn upload(&self) -> Option<ImageData<Rgba8>> {
        ImageData::new(self.width, self.height, Arc::clone(&self.data)).ok()
    }

    /// Whether `other` is this very grid — the same allocation at the same
    /// size — rather than merely equal pixels.
    pub(crate) fn is(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && Arc::ptr_eq(&self.data, &other.data)
    }
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

pub fn u32_to_f32(value: u32) -> f32 {
    value
        .to_f32()
        .expect("image dimensions must be representable as f32")
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

/// Where `pixels` land inside a `width` x `height` box, and whether the
/// placement needs the box clip. `None` when either side has no area — there
/// is no meaningful scale between them and the transform would be degenerate.
fn placement(
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

/// Records `image` across a `width` x `height` box, placed according to `mode`.
///
/// `Draw::image` takes the destination rectangle directly, so the scale and
/// centring the old API hid inside a transform live in the rect. When the
/// placement overflows the box, the draw runs inside a clip scope — the box
/// `Rect` itself, recorded as the shape it is rather than lowered to a path.
///
/// Records nothing when either the image or the box has no area.
pub fn draw(
    recorder: &mut Recorder,
    image: ImageId,
    pixels: (u32, u32),
    sampling: Sampling,
    mode: Option<ContentMode>,
    width: f32,
    height: f32,
) {
    let Some((destination, clipped)) = placement(pixels, mode, width, height) else {
        return;
    };
    if clipped {
        let bounds = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
        recorder.clip(bounds, |recorder| {
            recorder.image(image, destination, sampling);
        });
    } else {
        recorder.image(image, destination, sampling);
    }
}

/// Scene content that draws one decoded image for the lifetime of a mount.
///
/// The pixel grid uploads to the engine on the first `build_scene` — the
/// frame that first names its `ImageId`, as the registration contract asks —
/// and the [`Registered`] handle is what every later recording keeps naming.
/// The upload lives exactly as long as the content holds the handle.
///
/// A refused upload is settled the same way: the content keeps the
/// [`RegistrationError`], draws nothing, and does not offer the grid to that
/// engine again.
pub struct ImageSceneContent {
    pixels: Pixels,
    sampling: Sampling,
    mode: Option<ContentMode>,
    /// What the engine made of `pixels`, settled inside the first
    /// `build_scene`. `None` until that frame, or permanently when the grid
    /// has no pixels to upload.
    registration: Option<Registration>,
    /// The view's registration state, which this content reports its
    /// registration's outcome into. `None` for content rendered on its own,
    /// whose caller reads [`Self::rejection`] instead.
    state: Option<Binding<RegistrationState>>,
}

impl ImageSceneContent {
    pub const fn new(
        pixels: Pixels,
        sampling: Sampling,
        mode: Option<ContentMode>,
        state: Option<Binding<RegistrationState>>,
    ) -> Self {
        Self {
            pixels,
            sampling,
            mode,
            registration: None,
            state,
        }
    }

    /// The engine's refusal of this content's pixels, once a `build_scene`
    /// has offered them and been refused.
    pub fn rejection(&self) -> Option<&RegistrationError> {
        self.registration.as_ref()?.as_ref().err()
    }
}

impl fmt::Debug for ImageSceneContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImageSceneContent")
            .field("width", &self.pixels.width)
            .field("height", &self.pixels.height)
            .field("mode", &self.mode)
            .field("registration", &self.registration)
            .finish_non_exhaustive()
    }
}

impl SceneContent for ImageSceneContent {
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        if self.registration.is_none() {
            let Some(data) = self.pixels.upload() else {
                return false;
            };
            let registration = register(data, resources);
            if let Some(state) = &self.state {
                report(state, registration.as_ref().err(), || true);
            }
            self.registration = Some(registration);
        }
        let Some(Ok(image)) = &self.registration else {
            return false;
        };
        let id = resources.name(image);
        draw(
            recorder,
            id,
            (self.pixels.width, self.pixels.height),
            self.sampling,
            self.mode,
            width,
            height,
        );
        false
    }

    // The registration belongs to the engine that settled it, and so does a
    // refusal: the replacement engine offers `pixels` again on the next
    // `build_scene`, which is where registrations live anyway.
    fn rebuild_for_engine(&mut self) {
        self.registration = None;
    }

    fn intrinsic_size(&self) -> Option<LayoutSize> {
        pixel_size(self.pixels.width, self.pixels.height)
    }
}

#[cfg(test)]
pub mod test_backend {
    //! A scene backend whose image budget a test sets, and a frame driver
    //! over it, so a refused registration is reproducible without a GPU.

    use alloc::format;
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::Cell;

    use waterui_graphics::draw::{Command, Content, FontId, ImageId, ImageLimits, LayoutSize};
    use waterui_graphics::{
        FontSource, Handle, ImageData, ResourceError, ResourceHandle, Rgba8, Rgba16F, SceneBackend,
        SceneContent, SceneResources,
    };

    /// A texture the test backend handed out.
    struct Texture(ImageId);

    impl ResourceHandle for Texture {
        type Id = ImageId;

        fn id(&self) -> ImageId {
            self.0
        }
    }

    /// A backend that admits every size but holds at most `capacity` texels in
    /// all, refusing a registration the remaining budget cannot take — the
    /// way an engine refuses an image that fits its limits alone but not
    /// beside the images already registered.
    pub struct Budget {
        capacity: u64,
        /// Texels registered so far. Registrations are never released here:
        /// a test's images live for the whole test.
        used: Cell<u64>,
        /// Every registration that reached the backend, accepted or not.
        pub attempts: Cell<u32>,
    }

    impl Budget {
        pub fn new(capacity: u64) -> Rc<Self> {
            Rc::new(Self {
                capacity,
                used: Cell::new(0),
                attempts: Cell::new(0),
            })
        }

        fn admit(&self, width: u32, height: u32) -> Result<Handle<ImageId>, ResourceError> {
            self.attempts.set(self.attempts.get() + 1);
            let texels = u64::from(width) * u64::from(height);
            let used = self.used.get() + texels;
            if used > self.capacity {
                return Err(ResourceError::Image(format!(
                    "{width}x{height} does not fit the remaining image budget"
                )));
            }
            self.used.set(used);
            Ok(Handle::new(Texture(ImageId::new(u64::from(
                self.attempts.get(),
            )))))
        }
    }

    impl SceneBackend for Budget {
        fn register_font(&self, _source: FontSource) -> Result<Handle<FontId>, ResourceError> {
            Err(ResourceError::Unsupported("fonts"))
        }

        fn register_rgba8(&self, data: ImageData<Rgba8>) -> Result<Handle<ImageId>, ResourceError> {
            self.admit(data.width(), data.height())
        }

        fn register_rgba16f(
            &self,
            data: ImageData<Rgba16F>,
        ) -> Result<Handle<ImageId>, ResourceError> {
            self.admit(data.width(), data.height())
        }

        fn image_limits(&self) -> ImageLimits {
            ImageLimits::UNLIMITED
        }
    }

    /// Records one frame of `content` against `resources`: the image ids the
    /// recording names.
    pub fn frame(content: &mut impl SceneContent, resources: &SceneResources) -> Vec<ImageId> {
        let mut recording = resources.recording();
        let mut recorded = Content::record(&LayoutSize::new(), |recorder| {
            content.build_scene(recorder, &mut recording, 8.0, 8.0);
        });
        let _held = recording.finish();
        recorded
            .snapshot()
            .commands()
            .iter()
            .filter_map(|command| match command {
                Command::Image { image, .. } => Some(*image),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageSceneContent, LayoutSize, Pixels, destination, draw, overflows};
    use alloc::vec;
    use waterui_graphics::SceneContent as _;
    use waterui_graphics::draw::kurbo::{Rect, Size};
    use waterui_graphics::draw::{Command, Content, ImageId, Sampling};
    use waterui_layout::ContentMode;

    /// A 4:1 image in a square box: the two aspect ratios disagree, so every
    /// mode resolves to a different rectangle.
    const WIDE: Size = Size::new(80.0, 20.0);
    const SQUARE: Size = Size::new(100.0, 100.0);

    /// A registered image's place in a recording is its `ImageId`; a fixed raw
    /// value records identically to one a real engine minted.
    const IMAGE: ImageId = ImageId::new(1);

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

    /// The image commands a draw records, as `(destination, sampling)` pairs,
    /// plus a count of clip scopes.
    struct Recorded {
        images: Vec<(Rect, Sampling)>,
        clips: usize,
    }

    fn record(pixels: (u32, u32), mode: Option<ContentMode>, width: f32, height: f32) -> Recorded {
        let mut content = Content::record(&waterui_graphics::draw::LayoutSize::new(), |recorder| {
            draw(
                recorder,
                IMAGE,
                pixels,
                Sampling::Linear,
                mode,
                width,
                height,
            );
        });
        let mut recorded = Recorded {
            images: Vec::new(),
            clips: 0,
        };
        for command in content.snapshot().commands() {
            match command {
                Command::Image { dst, sampling, .. } => recorded.images.push((*dst, *sampling)),
                Command::BeginClip { .. } => recorded.clips += 1,
                _ => {}
            }
        }
        recorded
    }

    #[test]
    fn stretch_maps_the_pixel_grid_onto_the_whole_box() {
        let recorded = record((80, 20), None, 100.0, 100.0);
        assert_eq!(recorded.clips, 0);
        let [(destination, _)] = recorded.images[..] else {
            panic!("stretching must draw exactly one image");
        };
        // The image's own corners land on the box's corners.
        assert_eq!(destination, Rect::new(0.0, 0.0, 100.0, 100.0));
    }

    #[test]
    fn fill_clips_the_overflow_and_fit_does_not() {
        let filled = record((80, 20), Some(ContentMode::Fill), 100.0, 100.0);
        assert_eq!((filled.clips, filled.images.len()), (1, 1));
        // 5x scale, centred: the left edge starts 150 points off the box.
        assert_eq!(filled.images[0].0, Rect::new(-150.0, 0.0, 250.0, 100.0));

        let fitted = record((80, 20), Some(ContentMode::Fit), 100.0, 100.0);
        assert_eq!((fitted.clips, fitted.images.len()), (0, 1));
        assert_eq!(fitted.images[0].0, Rect::new(0.0, 37.5, 100.0, 62.5));
    }

    #[test]
    fn a_degenerate_box_or_image_draws_nothing() {
        assert_eq!(record((80, 20), None, 0.0, 100.0).images, []);
        assert_eq!(record((0, 0), None, 100.0, 100.0).images, []);
    }

    /// The natural size is the pixel grid at one pixel per unit, whatever the
    /// content mode: the mode decides where the pixels land inside a box, not
    /// how big the box wants to be.
    #[test]
    fn an_image_is_its_pixel_grid() {
        for mode in [None, Some(ContentMode::Fit), Some(ContentMode::Fill)] {
            let pixels = Pixels::new(vec![255; 80 * 20 * 4], 80, 20);
            let content = ImageSceneContent::new(pixels, Sampling::Linear, mode, None);
            assert_eq!(content.intrinsic_size(), Some(LayoutSize::new(80.0, 20.0)));
        }
    }

    /// An image with no pixels has no size of its own.
    #[test]
    fn an_empty_image_has_no_size() {
        assert_eq!(
            ImageSceneContent::new(Pixels::new(vec![], 0, 0), Sampling::Linear, None, None)
                .intrinsic_size(),
            None
        );
        assert_eq!(
            ImageSceneContent::new(Pixels::new(vec![], 80, 0), Sampling::Linear, None, None)
                .intrinsic_size(),
            None
        );
    }
}
