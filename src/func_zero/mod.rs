use crate::{
    base_func::{SessionId, UnexpectedError, CheatOrUnexpectedError},
    party::PartyId,
    field::{Group, RandElement},
};

pub trait AsyncZeroShare {
    async fn init(&self, sid: SessionId, parties: &[PartyId]) -> Result<(), CheatOrUnexpectedError>;
    fn generate_noninteractive<G: Group + RandElement>(&self, sid: SessionId) -> Result<G, UnexpectedError>;
}

pub mod folklore_ro_zero;
pub use folklore_ro_zero::FolkloreRoZeroPlayer;

