use crate::{
    base_func::{SessionId, UnexpectedError},
    field::RandElement,
    party::PartyId,
};

pub trait AsyncEot {
    /// Runs the setup to create the base OTs and correlation
    async fn init(
        &self,
        sid: SessionId,
        other: PartyId,
        is_sender: bool,
    ) -> Result<(), UnexpectedError>;

    /// As the sender send \vec{alpha}, and receive \vec{omega}, such that the receiver learns \vec{omega} + \vec{beta} * \vec{alpha}
    async fn send<T: RandElement>(
        &self,
        sid: SessionId,
        other: PartyId,
        num: usize
    ) -> Result<Vec<[T;2]>, UnexpectedError>;

    /// As the receiver send \vec{beta}, and receive \vec{omega} + \vec{beta} * \vec{alpha}
    async fn recv<T: RandElement>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: &[bool],
    ) -> Result<Vec<T>, UnexpectedError>;
}


pub mod zzzr23_eot;
pub use zzzr23_eot::ZzzrEotPlayer;
