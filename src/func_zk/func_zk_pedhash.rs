use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    field::{ExpGroup, Field, Group, RandElement},
    func_net::AsyncNet,
    func_vole::AsyncVole,
    party::PartyId,
};

use std::{marker::PhantomData, sync::Arc};

use anyhow::Context;

use log::trace;

pub trait AsyncZkPedHash {
    /// Start a new instance with `sid`
    async fn init(
        &self,
        sid: SessionId,
        prover: PartyId,
        verifier: PartyId,
    ) -> Result<(), UnexpectedError>;

    async fn prove<
        E: Field,
        G: Group + RandElement + ExpGroup<Ford = E>,
        H: Group + ExpGroup<Ford = E>,
        Hom: Fn(G) -> H,
    >(
        &self,
        sid: SessionId,
        hom: Hom,
        x: H,
        w: G,
    ) -> Result<(), CheatOrUnexpectedError>;
    async fn verify<
        E: Field,
        G: Group + RandElement + ExpGroup<Ford = E>,
        H: Group + ExpGroup<Ford = E>,
        Hom: Fn(G) -> H,
    >(
        &self,
        sid: SessionId,
        hom: Hom,
    ) -> Result<H, CheatOrUnexpectedError>;
}
