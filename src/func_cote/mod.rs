//! This module defines Oblivious Transfer related functionalities

use crate::{
    base_func::{SessionId, CheatOrUnexpectedError, UnexpectedError},
    field::{Ring, RandElement},
    party::PartyId,
    linalg::Vector,
};


/// Trait for Oblivious Transfer (extension)
///
///     --------   
///     |      | 
/// m_* |  OT  | b, m_b
/// <-- |      | -->  
///     --------
/// Where a sender learns two messages, and the receiver gets exactly one of them.
pub trait AsyncOt {
    /// Runs the setup to create the base OTs and correlation
    async fn init(
        &self,
        sid: SessionId,
        other: PartyId,
        is_sender: bool,
    ) -> Result<(), UnexpectedError>;

    /// As the sender send \vec{alpha}, and receive \vec{omega}, such that the receiver learns \vec{omega} + \vec{beta} * \vec{alpha}
    async fn rand_send<T: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<T, ELL>, Vector<T, ELL>), CheatOrUnexpectedError>;

    /// As the receiver send \vec{beta}, and receive \vec{omega} + \vec{beta} * \vec{alpha}
    async fn rand_recv<T: RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
    ) -> Result<(Vector<bool, ELL>, Vector<T, ELL>), CheatOrUnexpectedError>;
}

/// Trait for the Correlated OT (extension) functionality
///
///  a  --------  b
/// --> |      | <---
///  m  | COT  | m + ab
/// <-- |      | -->  
///     --------
/// Where a sender provides a list of correlations, a receiver provides choice bits
/// the sender gets random output m, and the receiver learns m + ba
pub trait AsyncCote {
    /// Start a new instance with `other` and `sid`
    /// Runs the setup to create the base OTs and correlation
    async fn init(
        &self,
        sid: SessionId,
        other: PartyId,
        is_sender: bool,
    ) -> Result<(), UnexpectedError>;

    /// As the sender send \vec{alpha}, and receive \vec{omega}, such that the receiver learns \vec{omega} + \vec{beta} * \vec{alpha}
    async fn send<T: Ring>(
        &self,
        sid: SessionId,
        other: PartyId,
        correlations: Vec<T>,
    ) -> Result<Vec<T>, UnexpectedError>;

    /// Calculate as in send, but record the transcript with `trace_fn`
    async fn send_trace<T: Ring, F: FnMut(&[u8])>(
        &self,
        sid: SessionId,
        other: PartyId,
        correlations: Vec<T>,
        trace_fn: F,
    ) -> Result<Vec<T>, UnexpectedError>;

    /// As the receiver send \vec{beta}, and receive \vec{omega} + \vec{beta} * \vec{alpha}
    async fn recv<T: Ring>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: Vec<bool>,
    ) -> Result<Vec<T>, UnexpectedError>;

    /// Calculate as in recv, but record the transcript with `trace_fn`
    async fn recv_trace<T: Ring, F: FnMut(&[u8])>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: Vec<bool>,
        trace_fn: F,
    ) -> Result<Vec<T>, UnexpectedError>;
}

pub mod kos_cote;
pub use kos_cote::KosCotePlayer;

pub mod silent_cote;
pub use silent_cote::SilentCotePlayer;

pub mod softspoken_ote;
pub use softspoken_ote::SoftspokenOtePlayer;
