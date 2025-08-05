use crate::{
    base_func::{CheatOrUnexpectedError, SessionId, UnexpectedError},
    field::{ExpGroup, ConstInt, Group, RandElement},
    party::PartyId,
};

pub trait AsyncZkSigma {
    /// Start a new instance with `sid`
    async fn init(
        &self,
        sid: SessionId,
        prover: PartyId,
        verifier: PartyId,
    ) -> Result<(), UnexpectedError>;

    async fn prove<
        E: ConstInt + RandElement,
        G: Group + RandElement + ExpGroup<E>,
        H: Group + ExpGroup<E>,
        Hom: Fn(G) -> H,
    >(
        &self,
        sid: SessionId,
        other: PartyId,
        hom: Hom,
        x: H,
        w: G,
    ) -> Result<(), CheatOrUnexpectedError>;

    async fn verify<
        E: ConstInt + RandElement,
        G: Group + RandElement + ExpGroup<E>,
        H: Group + ExpGroup<E>,
        Hom: Fn(G) -> H,
    >(
        &self,
        sid: SessionId,
        other: PartyId,
        hom: Hom,
        x: H,
    ) -> Result<H, CheatOrUnexpectedError>;
}

pub mod maurer15_zk;
pub use maurer15_zk::MaurerZkPlayer;
