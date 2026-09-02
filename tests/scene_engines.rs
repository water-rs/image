//! Renders a real decoded image through the scene contract and writes the
//! result out to be looked at.
//!
//! An image is drawn by `Scene2D::draw_image` under a transform this crate
//! derives from the content mode, so the thing under test is the transform: a
//! wide source in a square box lands somewhere different for each mode, and the
//! only way to be sure it landed there is to render it and look.
//!
//! The PNGs go to `/tmp/waterui_scene_engines/` alongside the other components'
//! scene exports. The colour assertions are exact rather than approximate: each
//! sample sits well inside a flat region of the source image, so a transposed,
//! mirrored, shifted or unscaled blit misses it.
//!
//! What the assertions deliberately do *not* pin is which way up the surface
//! ends: on Vulkan the classic scene surface blits through a shaderloom-compiled
//! shader that inverts Y, so every Linux render of one is upside down
//! (water-rs/waterui#239, open). That is a property of the blit, not of the
//! transform under test, and the exports carry a corner marker so a reviewer
//! reading them sees the orientation for themselves. Every sample below is
//! chosen to mean the same thing either way up.
#![cfg(feature = "gpu")]

use std::path::Path;

use image::ImageEncoder as _;
use waterui_graphics::{
    GpuRuntime, OffscreenRenderConfig, OffscreenRenderOutput, OffscreenSize, SceneEngine, wgpu,
};
use waterui_image::{ContentMode, Image};

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

/// Renders `image` into a `BOX_SIDE` square through the classic scene engine.
///
/// The offscreen surface *is* the box here, so the content mode is what places
/// the image in it; `.resizable()` belongs to the layout wrapper a view body
/// builds and has nothing to say about a surface of a fixed size.
///
/// The engine is pinned rather than left to the adapter so the export is the
/// same picture on every machine that runs the test.
fn render(image: Image) -> OffscreenRenderOutput {
    let size = OffscreenSize::try_from_pixels(BOX_SIDE, BOX_SIDE).expect("test size must be valid");
    let config = OffscreenRenderConfig::new(size)
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .scene_engine(SceneEngine::Classic);
    let runtime = pollster::block_on(GpuRuntime::new())
        .expect("scene image export requires a working GPU runtime");
    let mut env = waterui_core::Environment::new();
    pollster::block_on(image.render_offscreen(&runtime, config, &mut env))
        .expect("offscreen render should succeed")
}

fn save(output: &OffscreenRenderOutput, name: &str) {
    let directory = Path::new("/tmp/waterui_scene_engines");
    std::fs::create_dir_all(directory).expect("output directory must be creatable");
    output
        .save_png(directory.join(name))
        .expect("png should be written");
}

/// The pixel at `(x, y)` of a render.
fn pixel(output: &OffscreenRenderOutput, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * output.width + x) * 4) as usize;
    output.rgba8[offset..offset + 4]
        .try_into()
        .expect("a pixel is four bytes")
}

fn decoded_source() -> Image {
    Image::from_encoded(&source_png()).expect("the fixture PNG must decode")
}

#[test]
fn stretching_maps_the_whole_source_onto_the_whole_box() {
    let output = render(decoded_source());
    save(&output, "image_stretch.png");

    // Each source quarter becomes a full-height stripe 64 pixels wide.
    assert_eq!(pixel(&output, 32, 128), RED);
    assert_eq!(pixel(&output, 96, 128), GREEN);
    assert_eq!(pixel(&output, 160, 128), BLUE);
    assert_eq!(pixel(&output, 224, 128), WHITE);
}

#[test]
fn fit_keeps_the_aspect_ratio_and_leaves_the_box_empty_around_it() {
    let output = render(decoded_source().content_mode(ContentMode::Fit));
    save(&output, "image_fit.png");

    // 2x scale, so a 256x64 band centred vertically between y=96 and y=160.
    assert_eq!(pixel(&output, 32, 128), RED);
    assert_eq!(pixel(&output, 96, 128), GREEN);
    assert_eq!(pixel(&output, 160, 128), BLUE);
    assert_eq!(pixel(&output, 224, 128), WHITE);
    // Nothing is drawn above or below the band.
    assert_eq!(pixel(&output, 128, 32), TRANSPARENT);
    assert_eq!(pixel(&output, 128, 224), TRANSPARENT);
}

#[test]
fn fill_covers_the_box_and_clips_what_hangs_off_it() {
    let output = render(decoded_source().content_mode(ContentMode::Fill));
    save(&output, "image_fill.png");

    // 8x scale, so 1024x256 centred: the visible window is source x 48..80,
    // which is the second half of the green quarter and the first half of the
    // blue one, meeting at screen x=128.
    assert_eq!(pixel(&output, 64, 128), GREEN);
    assert_eq!(pixel(&output, 192, 128), BLUE);
    // The box is covered top to bottom, unlike `Fit`. The source's two-pixel
    // black border scales to sixteen, so these sit just inside it.
    assert_eq!(pixel(&output, 64, 24), GREEN);
    assert_eq!(pixel(&output, 192, 232), BLUE);
    // The red quarter and the corner marker are off the left edge entirely.
    assert!(
        !output.rgba8.as_chunks::<4>().0.contains(&MAGENTA),
        "the clip must keep the overflowing corner marker out of the box"
    );
}
