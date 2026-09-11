//! Image primitives and decode helpers for `WaterUI`.

extern crate alloc;

/// Image decode routing and HEIF compatibility helpers.
pub mod codec;
mod image;
mod scene;

pub use codec::DecodePath;
pub use image::{Image, Interpolation, ReactiveImage, ReactiveImageHandle, image, reactive_image};
/// How an image fills the box its layout gives it, re-exported so callers of
/// [`Image::content_mode`] need no direct dependency on the layout crate.
pub use waterui_layout::ContentMode;
