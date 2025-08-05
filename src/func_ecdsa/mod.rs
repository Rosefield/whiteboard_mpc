use crate::{
    party::PartyId,
    base_func::{SessionId, CheatOrUnexpectedError},
    ecgroup::{EcGroup, ReduceFromInteger},
    field::{Field, RandElement}
};

use sha2::{Sha256, Digest};

#[derive(Debug, PartialEq, Clone)]
pub struct EcdsaSignature<E: EcGroup> {
    pub s: E::Ford,
    pub rx: E::Ford,
}

impl<E: EcGroup> EcdsaSignature<E> {
    pub fn is_valid_for(&self, pk: &E, m: &[u8]) -> bool {
        let hm = Sha256::digest(m);
        let gen = E::one();
        let e: E::Ford = <E::Ford as ReduceFromInteger>::reduce(&hm);
        let rprime: E = (gen.exp(&e) + pk.exp(&self.rx)).exp(&self.s.inv().unwrap());
        let rpx = rprime.x_reduced();
        return rpx == self.rx;
    }
}

pub trait AsyncThreshEcdsa {
    type KeyShare<E: EcGroup>;
    async fn setup<E: EcGroup>(&self, sid: SessionId, parties: &[PartyId], threshold: usize) -> Result<Self::KeyShare<E>, CheatOrUnexpectedError>
        where E::Ford: RandElement;

    async fn sign<E: EcGroup>(&self, sid: SessionId, parties: &[PartyId], sigid: usize, share: Self::KeyShare<E>, msg: &[u8]) -> Result<EcdsaSignature<E>, CheatOrUnexpectedError>
        where E::Ford: RandElement;
}

pub mod dkls23_ecdsa;
pub use dkls23_ecdsa::Dkls23EcdsaPlayer;
