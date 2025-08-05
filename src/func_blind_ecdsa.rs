use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    field::{FWrap, RandElement, Ring},
    ecgroup::EcGroup,
    func_net::AsyncNet,
    func_vole::AsyncVole,
    party::PartyId,
};

use std::{marker::PhantomData, sync::Arc};

use anyhow::Context;

use log::trace;

use sha2::{Digest, Sha256};


pub struct DLRsBlindEcdsa<FN, FV, FZS, FZC, G> {
    party_id: PartyId,
    n: usize,
    net: Arc<FN>,
    vole: Arc<FV>,
    zksigma: Arc<FZS>,
    zkcircuit: Arc<FZC>,
    _g: PhantomData<G>,
}


pub trait AsyncBlindEcdsa<G: EcGroup> {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId, client: PartyId, server: PartyId) -> Result<(), UnexpectedError>;

    async fn sign_client(&self, sid: SessionId, m: &[u8]) -> Result<(G::Fq, G::Fq), CheatOrUnexpectedError>;
    async fn sign_server(&self, sid: SessionId) -> Result<(), CheatOrUnexpectedError>;
}


fn hom_pedersen<E: Field, G: Group + ExpGroup<Ford=E>>(g1: G, h1: G) -> impl Fn((E,E)) -> G {
    |es| {
        g1.clone().exp(es.0) + g2.clone().exp(es.0)
    }
}

fn rel_pedersen_2<E: Field + ExpGroup<Ford=E>, G: Group + ExpGroup<Ford=E>>(g1: G, h1: G) -> impl Fn(FWrap((FWrap((E,E)),FWrap((E,E))))) -> FWrap((G,G)) {
    let hom = hom_pedersen::<E,G>(g1, h1);

    |es| {
        FWrap((hom(es.0.0), hom(es.0.1)))
    }
}

fn rel_pedersen_sq<E: Field + ExpGroup<Ford=E>, G: Group + ExpGroup<Ford=E>>(g1: G, h1: G, c1: G) -> impl Fn(FWrap((E,E,E,E,E))) -> FWrap((G,G,G)) {
    let hom = hom_pedersen::<E,G>(g1, h1.clone());
    let hom2 = hom_pedersen::<E,G>(c1, h1);

    |es| {
        // should be able to remove the c2,t2 entirely?
        // (c1,t1, c2,t2, t2-c1t1) => (C1, C2, C2)
        FWrap((hom(FWrap((es.0.0, es.0.1))), hom(FWrap((es.0.2, es.0.3))), hom2(FWrap((es.0.0, es.0.4)))))
    }
}


fn rel_pedersen_prod<E: Field + ExpGroup<Ford=E>, G: Group + ExpGroup<Ford=E>>(g1: G, h1: G, c1: G) -> impl Fn(FWrap((E,E,E,E,E,E))) -> FWrap((G,G,G)) {
    let hom = hom_pedersen::<E,G>(g1, h1.clone());
    let hom2 = hom_pedersen::<E,G>(c1, h1);

    |es| {
        // (c1,t1, c2,t2, t2-c1t1) => (C1, C2, C2)
        FWrap((hom(FWrap((es.0.0, es.0.1))), hom(FWrap((es.0.2, es.0.3))), hom2(FWrap((es.0.0, es.0.4)))))
    }
}



impl<G: EcGroup, FN: AsyncNet, FV: AsyncVole, FZS: AsyncZkSigma, FZC: AsyncZkCircuit> AsyncBlindEcdsa<G> for DLRsBlindEcdsa<FN, FV, FZS, FZC, G> {


    async fn init(&self, sid: SessionId, client: PartyId, server: PartyId) -> Result<(), UnexpectedError> {
        unimplemented!()
    }

    async fn sign_client(&self, sid: SessionId, m: &[u8]) -> Result<(G::Fq, G::Fq), CheatOrUnexpectedError> {

        type Fq = G::Fq;
        type Fo = G::Ford;

        let (g1,h1) = {G::gen(), G::gen()};

        // round 1
        
        let FWrap((b4, phi_a, tm, tsha, ka)) = {
            let mut rng = rand::thread_rng();
            FWrap::<(Fo,Fo,Fo,Fo,Fo)>::rand(&mut rng)
        };

        
        let vole_sid = sid.derive_ssid(FuncId::Fblindecdsa);
        let vole1 = vole_sid.next();
        let vole2 = vole1.next();

        let sphi =  self.vole.input(vole1, phi_a).await?;
        let sb4 =  self.vole.input(vole2, b4).await?;

        let el_m = Fo::from_bytes(m);
        let hm = Sha2::new()
                .chain_update(m)
                .finalize();
        let el_sha = Fo::from_bytes(&hm);

        let commit = hom_pedersen(g1, h1);
        
        let cm = commit(el_m, tm);
        let csha = commit(el_sha, tsha);

        let _ = self.zkcircuit.prove(sid, (cm, csha), ((el_m, tm), (el_sha, tsha))).await?;

        let _ = self.pcom.init(sid, g1.exp(ka)).await?;
        
        // round 2

        let _ = self.pcom.share()




        

        let b4 = ();
        

    }

    async fn sign_server(&self, sid: SessionId) -> Result<(), CheatOrUnexpectedError> {

    }
}



