//! A `.resizable()` image has a size of its own: its pixel grid.
//!
//! These are the layout claims behind `SceneContent::intrinsic_size` for an
//! image, checked through the semantic runtime rather than by inspecting the
//! content: a vertical scroll view names the width and leaves the height open,
//! which is exactly the case that used to collapse to zero.

use hydrolysis_m3::install as install_m3;
use waterui::ViewExt as _;
use waterui::accessibility::AccessibilityRole;
use waterui::component::{hstack, text};
use waterui::layout::scroll::ScrollView;
use waterui_image::{ContentMode, Image};
use waterui_testing::{Role, SemanticApp, ui};

/// An 80 x 20 image: four times as wide as it is tall, so a wrong axis or a
/// stretched-to-viewport answer is a whole different number.
fn wide_image() -> Image {
    Image::new(vec![255; 80 * 20 * 4], 80, 20)
}

fn labelled(image: Image) -> impl waterui::View {
    image
        .a11y_role(AccessibilityRole::Image)
        .a11y_label("Wide image")
}

fn bounds(app: &mut SemanticApp) -> (f32, f32) {
    let bounds = app
        .query()
        .role(Role::IMAGE)
        .label("Wide image")
        .single()
        .bounds();
    (bounds.width(), bounds.height())
}

fn assert_close(actual: (f32, f32), expected: (f32, f32)) {
    assert!(
        (actual.0 - expected.0).abs() < 0.5 && (actual.1 - expected.1).abs() < 0.5,
        "expected {}x{}, got {}x{}",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    );
}

/// The scroll axis proposes nothing: the named width carries the pixel grid's
/// aspect ratio to the open height, 200 wide → 50 tall. The viewport is
/// deliberately shorter than that, so the scroll view's `max(content, viewport)`
/// cannot hide the answer.
#[test]
fn a_resizable_image_keeps_its_aspect_ratio_on_an_unconstrained_axis() {
    for mode in [None, Some(ContentMode::Fit), Some(ContentMode::Fill)] {
        let mut app = ui().theme(install_m3).viewport(200, 10).mount(move || {
            let image = wide_image().resizable();
            let image = match mode {
                Some(mode) => image.content_mode(mode),
                None => image,
            };
            ScrollView::vertical(labelled(image))
        });
        assert_close(bounds(&mut app), (200.0, 50.0));
    }
}

/// Given a box, a resizable image still fills it: the pixel grid is what layout
/// falls back to, never a cap on what a container may ask for.
#[test]
fn a_resizable_image_still_fills_a_frame() {
    let mut app = ui()
        .theme(install_m3)
        .viewport(400, 400)
        .mount(|| labelled(wide_image().resizable()).size(160.0, 90.0));
    assert_close(bounds(&mut app), (160.0, 90.0));
}

/// A non-resizable image is rigid at its pixel size whatever it is offered: in
/// a row it takes its own 80 x 20 and leaves the rest to its sibling.
#[test]
fn a_non_resizable_image_stays_at_its_pixel_size() {
    let mut app = ui().theme(install_m3).viewport(400, 200).mount(|| {
        hstack((
            labelled(wide_image()),
            text("beside it").a11y_label("beside it"),
        ))
    });
    assert_close(bounds(&mut app), (80.0, 20.0));
}
