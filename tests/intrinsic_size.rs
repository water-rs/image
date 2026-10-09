//! Images have a natural size: their pixel grid.
//!
//! These are the layout claims behind `SceneContent::intrinsic_size` for an
//! image, checked through the semantic runtime rather than by inspecting the
//! content: a vertical scroll view names the width and leaves the height open,
//! which is exactly the case that used to collapse to zero.

use hydrolysis_m3::Material3;
use waterui::accessibility::AccessibilityRole;
use waterui::component::{hstack, text};
use waterui::layout::{StretchAxis, scroll::ScrollView};
use waterui::{AnyView, View, ViewExt as _};
use waterui_image::{ContentMode, Image, reactive_image};
use waterui_testing::{OffscreenApp, Role, ui};

/// An 80 x 20 image: four times as wide as it is tall, so a wrong axis or a
/// stretched-to-viewport answer is a whole different number.
fn wide_image() -> Image {
    Image::new(vec![255; 80 * 20 * 4], 80, 20)
}

fn labelled(image: impl View) -> impl View {
    image
        .a11y_role(AccessibilityRole::Image)
        .a11y_label("Wide image")
}

fn bounds(app: &mut OffscreenApp) -> (f32, f32) {
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
        (actual.0 - expected.0).abs() < 0.01 && (actual.1 - expected.1).abs() < 0.01,
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
        let mut app = ui()
            .theme(Material3::defaults())
            .viewport(200, 10)
            .mount_offscreen(move || {
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
    for reactive in [false, true] {
        for mode in [None, Some(ContentMode::Fit), Some(ContentMode::Fill)] {
            let mut app = ui()
                .theme(Material3::defaults())
                .viewport(400, 400)
                .mount_offscreen(move || {
                    let image = wide_image().resizable();
                    let image = match mode {
                        Some(mode) => image.content_mode(mode),
                        None => image,
                    };
                    let view = if reactive {
                        let (handle, view) = reactive_image();
                        handle.set(image);
                        let view = view.resizable();
                        AnyView::new(match mode {
                            Some(mode) => view.content_mode(mode),
                            None => view,
                        })
                    } else {
                        AnyView::new(image)
                    };
                    labelled(view).size(160.0, 90.0)
                });
            assert_close(bounds(&mut app), (160.0, 90.0));
        }
    }
}

/// A non-resizable image never grows: in a roomy row it takes its own 80 x 20
/// and leaves the rest to its sibling.
#[test]
fn a_non_resizable_image_stays_at_its_pixel_size() {
    let mut app = ui()
        .theme(Material3::defaults())
        .viewport(400, 200)
        .mount_offscreen(|| {
            hstack((
                labelled(wide_image()),
                text("beside it").a11y_label("beside it"),
            ))
        });
    assert_close(bounds(&mut app), (80.0, 20.0));
}

#[test]
fn non_resizable_images_only_scale_down() {
    for reactive in [false, true] {
        for (viewport, open_axes, expected) in [
            ((1, 1), 2, (640.0, 360.0)),
            ((480, 1), 1, (480.0, 270.0)),
            ((1000, 1), 1, (640.0, 360.0)),
            ((100, 100), 0, (100.0, 56.25)),
        ] {
            let mut app = ui()
                .theme(Material3::defaults())
                .viewport(viewport.0, viewport.1)
                .mount_offscreen(move || {
                    let image = Image::new(vec![255; 640 * 360 * 4], 640, 360);
                    let view = if reactive {
                        let (handle, view) = reactive_image();
                        handle.set(image);
                        AnyView::new(view)
                    } else {
                        AnyView::new(image)
                    };
                    assert_eq!(view.stretch_axis(), StretchAxis::None);
                    let view = labelled(view);
                    match open_axes {
                        2 => AnyView::new(ScrollView::both(view)),
                        1 => AnyView::new(ScrollView::vertical(view)),
                        _ => AnyView::new(view),
                    }
                });
            assert_close(bounds(&mut app), expected);
        }
    }
}

#[test]
fn a_reactive_image_remeasures_when_its_pixels_change() {
    let (handle, view) = reactive_image();
    let view = std::cell::RefCell::new(Some(view));
    let mut app = ui()
        .theme(Material3::defaults())
        .viewport(480, 1)
        .mount_offscreen(move || {
            ScrollView::vertical(labelled(
                view.borrow_mut().take().expect("image is mounted once"),
            ))
        });
    let empty_bounds = bounds(&mut app);

    handle.set(Image::new(vec![255; 640 * 360 * 4], 640, 360));
    app.settle();
    assert_close(bounds(&mut app), (480.0, 270.0));

    handle.set(wide_image());
    app.settle();
    assert_close(bounds(&mut app), (80.0, 20.0));

    handle.clear();
    app.settle();
    assert_close(bounds(&mut app), empty_bounds);
}
