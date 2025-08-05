use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    field::{ConstInt, ExpGroup, Group, RandElement},
    func_net::AsyncNet,
    func_zk::AsyncZkSigma,
    party::PartyId,
};

use anyhow::Context;

pub struct MaurerZkPlayer<FN> {
    party_id: PartyId,
    net: FN,
}

impl<FN> BaseFunc for MaurerZkPlayer<FN> {
    const FUNC_ID: FuncId = FuncId::Fzk;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

impl<FN: AsyncNet> AsyncZkSigma for MaurerZkPlayer<FN> {
    /// Start a new instance with `sid`
    async fn init(
        &self,
        _sid: SessionId,
        _prover: PartyId,
        _verifier: PartyId,
    ) -> Result<(), UnexpectedError> {
        unimplemented!()
    }

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
        _x: H,
        w: G,
    ) -> Result<(), CheatOrUnexpectedError> {
        let mut rng = rand::thread_rng();

        let mask = G::rand(&mut rng);
        let commit = hom(mask.clone());

        let mut buf = vec![0u8; H::BYTES];
        commit.to_bytes(&mut buf);

        let _ = self
            .net
            .send_to_local(other, FuncId::Fzk, sid, &buf)
            .await
            .with_context(|| self.err(sid, "failed to send zk commit randomness"))?;

        let mut cbuf = vec![0u8; E::BYTES];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fzk, sid, &mut cbuf)
            .await
            .with_context(|| self.err(sid, "failed to recv zk challenge"))?;
        let challenge = E::from_bytes(&cbuf);

        let response = mask + w.exp(&challenge);
        let mut rbuf = vec![0u8; G::BYTES];
        response.to_bytes(&mut rbuf);
        let _ = self
            .net
            .send_to_local(other, FuncId::Fzk, sid, &rbuf)
            .await
            .with_context(|| self.err(sid, "failed to send zk response"))?;

        Ok(())
    }

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
    ) -> Result<H, CheatOrUnexpectedError> {
        let mut buf = vec![0u8; H::BYTES];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fzk, sid, &mut buf)
            .await
            .with_context(|| self.err(sid, "failed to recv zk commit randomness"))?;
        let commit = H::from_bytes(&buf);

        let mut rng = rand::thread_rng();
        let challenge = E::rand(&mut rng);

        let mut cbuf = vec![0u8; E::BYTES];
        challenge.to_bytes(&mut cbuf);

        let _ = self
            .net
            .send_to_local(other, FuncId::Fzk, sid, &cbuf)
            .await
            .with_context(|| self.err(sid, "failed to send zk challenge"))?;
        let challenge = E::from_bytes(&cbuf);

        let mut rbuf = vec![0u8; G::BYTES];
        let _ = self
            .net
            .recv_from_local(other, FuncId::Fzk, sid, &mut rbuf)
            .await
            .with_context(|| self.err(sid, "failed to recv zk response"))?;
        let response = G::from_bytes(&rbuf);

        if hom(response) == commit + x.exp(&challenge) {
            Ok(x)
        } else {
            Err(self
                .cheat(sid, Some(other), "Invalid PoK".to_string())
                .into())
        }
    }
}

#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn test_pok_dl() {}
}
