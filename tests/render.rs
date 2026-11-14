//! Renders a real decoded image as a canvas drawing and writes the result out
//! to be looked at.
//!
//! An image is one `image` command at a destination rectangle this crate
//! derives from the content mode, so the thing under test is that rectangle: a
//! wide source in a square box lands somewhere different for each mode, and the
//! only way to be sure it landed there is to render it and look.
//!
//! The PNGs go under the test target's temporary directory. The colour
//! assertions are exact rather than approximate: each
//! sample sits well inside a flat region of the source image, so a transposed,
//! mirrored, shifted or unscaled blit misses it.
//!
//! The fixture's magenta corner marker pins orientation: the engine's readback
//! defines which way up the frame lands, and the marker lets a reviewer reading
//! the exports see it for themselves. Every sample below is chosen to mean the
//! same thing either way up.
#![cfg(feature = "gpu")]

use std::path::PathBuf;

use image::ImageEncoder as _;
use palette::color_difference::Ciede2000 as _;
use palette::{IntoColor as _, Lab, Srgb};
use waterui_core::Signal as _;
use waterui_graphics::canvas::ImageStatus;
use waterui_graphics::gpu::Gpu;
use waterui_graphics::{OffscreenImage, OffscreenRenderer, OffscreenSize};
use waterui_image::{ContentMode, Image, RenderOffscreenError};

/// Source width. Four times the height, so no mode agrees with any other in a
/// square box.
const SOURCE_WIDTH: u32 = 128;
/// Source height.
const SOURCE_HEIGHT: u32 = 32;
/// Side of the square box every render targets.
const BOX_SIDE: u32 = 256;

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const WHITE: [u8; 4] = [255, 255, 255, 255];
const MAGENTA: [u8; 4] = [255, 0, 255, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];
const TRANSPARENT: [u8; 4] = [0, 0, 0, 0];

/// A PNG whose every region is identifiable on sight.
///
/// Four colour quarters left to right, a magenta square in the top-left corner
/// and a black border: between them they pin down scale, position, orientation
/// and handedness.
fn source_png() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((SOURCE_WIDTH * SOURCE_HEIGHT * 4) as usize);
    for y in 0..SOURCE_HEIGHT {
        for x in 0..SOURCE_WIDTH {
            let quarter = x * 4 / SOURCE_WIDTH;
            let border = x < 2 || y < 2 || x >= SOURCE_WIDTH - 2 || y >= SOURCE_HEIGHT - 2;
            let colour = if border {
                BLACK
            } else if x < 8 && y < 8 {
                MAGENTA
            } else {
                match quarter {
                    0 => RED,
                    1 => GREEN,
                    2 => BLUE,
                    _ => WHITE,
                }
            };
            pixels.extend_from_slice(&colour);
        }
    }

    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            &pixels,
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            image::ExtendedColorType::Rgba8,
        )
        .expect("the fixture must encode as a PNG");
    png
}

/// Renders `image` into a `BOX_SIDE` square offscreen target.
///
/// The offscreen surface *is* the box here, so the content mode is what places
/// the image in it; `.resizable()` belongs to the layout wrapper a view body
/// builds and has nothing to say about a surface of a fixed size.
fn render(image: Image) -> OffscreenImage {
    try_render(image).expect("offscreen render should succeed")
}

fn try_render(image: Image) -> Result<OffscreenImage, RenderOffscreenError> {
    let renderer =
        OffscreenRenderer::<Gpu>::new().expect("scene image export requires a working GPU");
    let size = OffscreenSize::try_from_pixels(BOX_SIDE, BOX_SIDE).expect("test size must be valid");
    image.render_offscreen(&renderer, size, 1.0)
}

fn save(output: &OffscreenImage, name: &str) {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("render");
    std::fs::create_dir_all(&directory).expect("output directory must be creatable");
    output
        .save_png(directory.join(name))
        .expect("png should be written");
}

/// The largest CIEDE2000 difference that still reads as the expected colour:
/// one unit is the just-noticeable difference.
const JUST_NOTICEABLE: f32 = 1.0;

/// The colour of an 8-bit sRGB pixel in CIE L\*a\*b\*.
fn lab([r, g, b, _]: [u8; 4]) -> Lab {
    Srgb::new(r, g, b).into_format::<f32>().into_color()
}

/// Asserts that the pixel at `(x, y)` of a render reads as `expected`: its
/// alpha within one 8-bit step, and, where it is visible, its colour less
/// than a just-noticeable difference away.
#[track_caller]
fn assert_colour(output: &OffscreenImage, x: u32, y: u32, expected: [u8; 4]) {
    let actual = output.pixel(x, y);
    assert!(
        actual[3].abs_diff(expected[3]) <= 1,
        "pixel ({x}, {y}) is {actual:?}, expected the alpha of {expected:?}"
    );
    if expected[3] == 0 {
        return;
    }
    let difference = lab(actual).difference(lab(expected));
    assert!(
        difference <= JUST_NOTICEABLE,
        "pixel ({x}, {y}) is {actual:?}, {difference} CIEDE2000 from {expected:?}"
    );
}

fn decoded_source() -> Image {
    Image::from_encoded(&source_png()).expect("the fixture PNG must decode")
}

#[test]
fn stretching_maps_the_whole_source_onto_the_whole_box() {
    let output = render(decoded_source());
    save(&output, "image_stretch.png");

    // Each source quarter becomes a full-height stripe 64 pixels wide.
    assert_colour(&output, 32, 128, RED);
    assert_colour(&output, 96, 128, GREEN);
    assert_colour(&output, 160, 128, BLUE);
    assert_colour(&output, 224, 128, WHITE);
}

#[test]
fn fit_keeps_the_aspect_ratio_and_leaves_the_box_empty_around_it() {
    let output = render(decoded_source().content_mode(ContentMode::Fit));
    save(&output, "image_fit.png");

    // 2x scale, so a 256x64 band centred vertically between y=96 and y=160.
    assert_colour(&output, 32, 128, RED);
    assert_colour(&output, 96, 128, GREEN);
    assert_colour(&output, 160, 128, BLUE);
    assert_colour(&output, 224, 128, WHITE);
    // Nothing is drawn above or below the band.
    assert_colour(&output, 128, 32, TRANSPARENT);
    assert_colour(&output, 128, 224, TRANSPARENT);
}

#[test]
fn fill_covers_the_box_and_clips_what_hangs_off_it() {
    let output = render(decoded_source().content_mode(ContentMode::Fill));
    save(&output, "image_fill.png");

    // 8x scale, so 1024x256 centred: the visible window is source x 48..80,
    // which is the second half of the green quarter and the first half of the
    // blue one, meeting at screen x=128.
    assert_colour(&output, 64, 128, GREEN);
    assert_colour(&output, 192, 128, BLUE);
    // The box is covered top to bottom, unlike `Fit`. The source's two-pixel
    // black border scales to sixteen, so these sit just inside it.
    assert_colour(&output, 64, 24, GREEN);
    assert_colour(&output, 192, 232, BLUE);
    // The red quarter and the corner marker are off the left edge entirely.
    assert!(
        !output.rgba8().as_chunks::<4>().0.contains(&MAGENTA),
        "the clip must keep the overflowing corner marker out of the box"
    );
}

/// An image wider than any GPU's largest texture is refused by the engine at
/// registration: rendering it reports that refusal instead of panicking or
/// handing back a frame with nothing in it.
#[test]
fn an_image_the_engine_refuses_is_an_error_not_a_blank_frame() {
    const WIDTH: u32 = 1 << 17;
    let image = Image::new(vec![255; WIDTH as usize * 4], WIDTH, 1);
    let status = image.status();
    assert_eq!(status.snapshot(), ImageStatus::Pending);
    let error = try_render(image).expect_err("an image past the texture limit must be refused");
    let RenderOffscreenError::Rejected(error) = error else {
        panic!("expected the engine's refusal, got {error}");
    };
    let reason = error.to_string();
    assert!(
        reason.contains(&format!("image {WIDTH}x1 exceeds the image limits")),
        "{reason}"
    );
    // The same refusal is the image's status, which is how an app sees it.
    assert!(
        matches!(status.snapshot(), ImageStatus::Refused(refusal) if refusal.to_string() == reason),
        "the image's status carries the refusal"
    );
}

/// A drawn image reports that the renderer took it.
#[test]
fn a_drawn_image_is_ready() {
    let image = decoded_source();
    let status = image.status();
    render(image);
    assert_eq!(status.snapshot(), ImageStatus::Ready);
}
