//! The registration API that [`ImageStatus`] replaces.
//!
//! [`Image::registration`](crate::Image::registration) and
//! [`ReactiveImageHandle::registration`](crate::ReactiveImageHandle::registration)
//! stay for the one transition in which the framework's media component moves
//! to `status()`, and are then removed. Both read the same [`ImageStatus`] the
//! new API reads; new code uses `status()`.

use alloc::sync::Arc;
use core::fmt;

use waterui_graphics::ResourceError;
use waterui_graphics::canvas::ImageStatus;

/// Why the render target refused to register an image.
///
/// Cheap to clone: every clone shares the one backend error.
#[derive(Debug, Clone)]
pub struct RegistrationError(Arc<ResourceError>);

impl RegistrationError {
    /// The error the render target refused the registration with.
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

/// How the registration of an image's pixels with the render target stands.
#[derive(Debug, Clone)]
pub enum RegistrationState {
    /// The target has not tried to register the pixels yet.
    Pending,
    /// The target registered the pixels; the image is on screen.
    Registered,
    /// The target refused the pixels. The image draws nothing.
    Refused(RegistrationError),
}

impl From<ImageStatus> for RegistrationState {
    fn from(status: ImageStatus) -> Self {
        match status {
            ImageStatus::Pending => Self::Pending,
            ImageStatus::Ready => Self::Registered,
            ImageStatus::Refused(error) => Self::Refused(RegistrationError(error)),
            status => unreachable!("the registration API has no state for {status:?}"),
        }
    }
}
