//! Image primitives and decode helpers for `WaterUI`.

extern crate alloc;

/// Image decode routing and HEIF compatibility helpers.
pub mod codec;
mod image;
mod placement;
mod registration;

pub use codec::DecodePath;
#[cfg(feature = "gpu")]
pub use image::RenderOffscreenError;
pub use image::{Image, Interpolation, ReactiveImage, ReactiveImageHandle, image, reactive_image};
pub use registration::{RegistrationError, RegistrationState};
/// What a render target made of an image, re-exported so callers of
/// [`Image::status`] and [`ReactiveImageHandle::status`] need no direct
/// dependency on the graphics crate.
pub use waterui_graphics::canvas::ImageStatus;
/// How an image fills the box its layout gives it, re-exported so callers of
/// [`Image::content_mode`] need no direct dependency on the layout crate.
pub use waterui_layout::ContentMode;
