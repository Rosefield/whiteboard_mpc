use crate::{
    party::PartyId,
    base_func::{FuncId, BaseFunc, SessionId, UnexpectedError, CheatOrUnexpectedError},
    ecgroup::{EcGroup, ReduceFromInteger},
    field::{ConstInt, ToFromBytes, Field, RandElement, FWrap},
    linalg::Vector,
    func_ecdsa::{EcdsaSignature, AsyncThreshEcdsa},
    func_net::AsyncNet,
    func_vole::AsyncVole,
    func_com::AsyncCom,
    func_zero::AsyncZeroShare,
    common_protos::broadcast_opportunistic,
    polynomial::{InterpolationPolynomial, lagrange_poly},
};

use std::{
    sync::Arc,
    collections::HashMap,
};

use tokio::io;
        
use sha2::Digest;

use futures::{
    stream::FuturesUnordered,
    StreamExt,
};


pub struct Dkls23EcdsaPlayer<FN, FV, FC, FZ> {
    party_id: PartyId,
    net: Arc<FN>,
    vole: Arc<FV>,
    com: Arc<FC>,
    zero: Arc<FZ>
}

#[derive(Clone, Debug)]
pub struct KeyShare<E: EcGroup> {
    pub pk: E,
    pub sk_share: E::Ford,
    pub party_id: PartyId,
    pub all_parties: Vec<PartyId>,
    pub threshold: usize
}

impl<E: EcGroup> KeyShare<E> {
    fn additive_share(&self, subset: &[PartyId]) -> E::Ford {
        assert!(subset.len() >= self.threshold, "Insufficient parties to construct shares");
        assert!(subset.iter().all(|p| self.all_parties.contains(p)), "Can only convert to shares for subsets of the original parties");
        assert!(subset.contains(&self.party_id), "Can only convert my share if I am one of the target parties");

        let points: Vec<_> = subset.iter().map(|&p| E::Ford::from(p as u64)).collect();
        let xi = E::Ford::from(self.party_id as u64);

        let l = lagrange_poly(&points, &xi, |x| -x.clone());

        l * &self.sk_share
    }
}

impl<FN, FV, FC, FZ> BaseFunc for Dkls23EcdsaPlayer<FN, FV, FC, FZ> {
    const FUNC_ID: FuncId = FuncId::Fecdsa;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet, FuncId::Fvole, FuncId::Fcom];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

impl<FN: AsyncNet, FV: AsyncVole, FC: AsyncCom, FZ: AsyncZeroShare> AsyncThreshEcdsa for Dkls23EcdsaPlayer<FN, FV, FC, FZ> {
    type KeyShare<E: EcGroup> = KeyShare<E>;

    async fn setup<E: EcGroup>(&self, sid: SessionId, parties: &[PartyId], threshold: usize) -> Result<Self::KeyShare<E>, CheatOrUnexpectedError> 
        where E::Ford: RandElement
    {
        self.dlkeygen(sid, parties, threshold).await
    }

    async fn sign<E: EcGroup>(&self, sid: SessionId, parties: &[PartyId], sigid: usize, share: Self::KeyShare<E>, msg: &[u8]) -> Result<EcdsaSignature<E>, CheatOrUnexpectedError> 
        where E::Ford: RandElement
    {

        let FWrap((r_i, phi_i)): FWrap<(E::Ford, E::Ford)> = {
            let mut rng = rand::thread_rng();
            FWrap::rand(&mut rng)
        };

        let gen = E::one();
        let com_ri = gen.exp(&r_i);

        let mut chis = HashMap::new();

        let zerosid = sid.derive_ssid_context(FuncId::Fecdsa, &sigid);
        self.zero.init(zerosid, parties).await?;

        let zshare: E::Ford = self.zero.generate_noninteractive(zerosid)?;

        let sk_share = share.additive_share(parties) + zshare;
        let mul: Vector<E::Ford, 2> = [r_i.clone(), sk_share.clone()].into();

        enum Round1Msg<E: EcGroup> {
            ComVole(PartyId, (Result<(), UnexpectedError>, Result<(), UnexpectedError>, Result<Vector<E::Ford, 2>, CheatOrUnexpectedError>, Result<Vector<E::Ford, 2>, CheatOrUnexpectedError>))
        }

        // TODO: this should maybe depend on pi/pj instead of just next and swapping
        let comsid = sid.derive_ssid_context(FuncId::Fecdsa, &sigid);
        let comsid2 = comsid.next();

        let volesid = sid.derive_ssid_context(FuncId::Fecdsa, &sigid);
        let volesid2 = volesid.next();

        let netsid = sid.derive_ssid_context(FuncId::Fecdsa, &sigid);

        let mut round1_futs = FuturesUnordered::new();

        // Queue each of the pairwise party computations
        for other in parties.iter().filter(|p| **p != self.party_id) {

            let other = other.clone();
            let chi = {
                let mut rng = rand::thread_rng();
                E::Ford::rand(&mut rng)
            };
            chis.insert(other, chi.clone());

            round1_futs.push({
                // I really shouldn't have to clone here, but I can't figure out why the borrow
                // checker says that the reference to mul outlives, but not the reference to com_ri
                let mul = mul.clone();
                let com_ri = &com_ri;
                let [comsid, comsid2, volesid, volesid2] = if self.party_id < other {
                    [comsid, comsid2, volesid, volesid2]
                } else {
                    [comsid2, comsid, volesid2, volesid]

                };
                async move {
                    let mut bytes = vec![0u8; E::BYTES];
                    com_ri.to_bytes(&mut bytes);
                    let comf = self.com.commit_to(comsid, other, bytes);
                    let expf = self.com.expect_from(comsid2, other);

                    let inpf = self.vole.input::<_, 2>(volesid, other, chi);
                    let mulf = self.vole.multiply(volesid2, other, &mul);
                    Round1Msg::<E>::ComVole(other, tokio::join!(comf, expf, inpf, mulf))
                }
            });
        }

        let mut round2_futs = FuturesUnordered::new();


        let pk_i = gen.exp(&sk_share);

        enum Round2Msg<E: EcGroup> {
            ComCheck(PartyId, (Result<(), UnexpectedError>, Result<E, CheatOrUnexpectedError>, io::Result<()>, io::Result<FWrap<(E, E, E, E::Ford)>>)),
        }

        let mut ds_js = HashMap::new();
        let mut cs_js = HashMap::new();


        while let Some(res) = round1_futs.next().await {
            match res {
                Round1Msg::ComVole(other, (r1, r2, r3, r4)) => {
                    let (_, _, ds, cs) = (r1?, r2?, r3?, r4?);

                    ds_js.insert(other, ds);
                    cs_js.insert(other, cs.clone());
                    let [cu, cv] = cs.as_array();


                    let mut decbytes = vec![0u8; E::BYTES];
                    com_ri.to_bytes(&mut decbytes);
                    let gamma_ju = gen.exp(&cu);
                    let gamma_jv = gen.exp(&cv);
                    let psi_j: E::Ford = phi_i.clone() - &chis[&other];

                    let msg = FWrap((gamma_ju, gamma_jv, pk_i.clone(), psi_j));
                    let mut sendbytes = vec![0u8; msg.num_bytes()];
                    let recvbytes = sendbytes.clone();
                    msg.to_bytes(&mut sendbytes);

                    round2_futs.push({
                        let [comsid, comsid2] = if self.party_id < other {
                            [comsid, comsid2]
                        } else {
                            [comsid2, comsid]
                        };
                        async move { 
                            let decf = self.com.decommit_to(comsid, other, decbytes);
                            let valf = async { 
                                let r = self.com.value_from(comsid2, other, E::BYTES).await;
                                r.map(|bs| {E::from_bytes(&bs[..])})
                            };


                            let sendf = self.net.send_to_local(other, FuncId::Fecdsa, netsid, sendbytes);
                            let recvf = async {
                                let r = self.net.recv_from_local(other, FuncId::Fecdsa, netsid, recvbytes).await;
                                r.map(|(bs, _)| { FWrap::from_bytes(&bs) })
                            };

                            Round2Msg::ComCheck(other, tokio::join!(decf, valf, sendf, recvf))
                        }
                    });
                },
            }
        };

        let mut pk_prime = pk_i;
        let mut com_r = com_ri.clone();
        let mut phi = phi_i.clone();
        let mut cd: Vector<E::Ford, 2> = Vector::zero();

        // TODO: communicate failure instead of exiting immediately

        let mut cheat = None;


        while let Some(res) = round2_futs.next().await {
            match res {
                Round2Msg::ComCheck(other, (r1, r2, r3, r4)) => {
                    let (_, com_rj, _, check_j) = (r1?, r2?, r3?, r4?);

                    let FWrap((gamma_ju, gamma_jv, pk_j, psi_j)) = check_j;

                    let chi_j = &chis[&other];
                    let ds_j = &ds_js[&other];
                    let cs_j = &cs_js[&other];

                    let [du, dv] = ds_j.clone().as_array();
                    if (com_rj.exp(&chi_j) - gamma_ju) != gen.exp(&du) || 
                        (pk_j.exp(&chi_j) - gamma_jv) != gen.exp(&dv) {
                            cheat = Some(self.cheat(sid, Some(other), "invalid check adj message".to_string()));
                    }

                    pk_prime += pk_j;
                    com_r += com_rj;
                    phi += psi_j;
                    cd += cs_j;
                    cd += ds_j;
                }
            }
        }

        drop(round2_futs);

        if pk_prime != share.pk {
            cheat = Some(self.cheat(sid, None, "Mismatched public keys".to_string()));
        }

        let [c0, c1] = cd.as_array();

        let ui = r_i * &phi + c0;
        let vi = sk_share * &phi + c1;

        let rx = com_r.x_reduced();
        let hm = sha2::Sha256::digest(msg);
        let e = <E::Ford as ReduceFromInteger>::reduce(&hm);
        let wi: E::Ford = e*phi_i + rx.clone()*vi;

        let frag = FWrap([wi, ui]);
        let mut bytes = vec![0u8; frag.num_bytes()+1];

        enum _Round3Msg<E: EcGroup> {
            Fail,
            Fragment(FWrap<[E::Ford; 2]>)
        }
        if cheat.is_none() {
            bytes[0] = 1;
            frag.to_bytes(&mut bytes[1..]);
        }

        let results = broadcast_opportunistic(FuncId::Fecdsa, netsid, bytes, self.party_id, parties, self.net.clone()).await?;


        let frags = results.into_iter()
                        .map(|(k, v)| {
                            if v[0] == 0u8 {
                                Err(k)
                            } else {
                                Ok(FWrap::from_bytes(&v[1..]))
                            }
                        }).collect::<Result<Vec<FWrap<[E::Ford; 2]>>, PartyId>>();

        if cheat.is_some() || frags.is_err() {
            return Err(cheat.unwrap_or_else(|| self.cheat(sid, Some(frags.err().unwrap()), "Received fail message from another party".to_string())).into());
        }

        let FWrap([w, u]): FWrap<[E::Ford; 2]> = frag + frags.unwrap().into_iter().sum::<FWrap<[_; 2]>>();

        let s = w * u.inv().unwrap();

        let signature =  EcdsaSignature {
            s,
            rx
        };

        if !signature.is_valid_for(&share.pk, msg) {
            return Err(self.cheat(sid, None, "Failed to generate a valid signature".to_string()).into());
        }

        Ok(signature)
    }
}

impl<FN: AsyncNet, FV: AsyncVole, FC: AsyncCom, FZ> Dkls23EcdsaPlayer<FN, FV, FC, FZ> {
    async fn dlkeygen<E: EcGroup>(&self, sid: SessionId, parties: &[PartyId], threshold: usize) -> Result<KeyShare<E>, CheatOrUnexpectedError> 
        where E::Ford: RandElement
    {
        let Some(my_idx) = parties.iter().position(|p| *p == self.party_id) else {
            return Err(self.unexpected(sid, "Current party not one of the expected parties").into());
        };

        let my_idx = my_idx + 1;

        let mut points = vec![E::Ford::zero()];
        points.extend(parties.iter().map(|p| E::Ford::from(*p as u64)));

        let shares = {
            let mut rng = rand::thread_rng();
            InterpolationPolynomial::rand_share(&mut rng, threshold, &points)
        };
        let gen = E::one();
        // commit in the exponent to t points
        let point_coms: Vec<E> = shares.vals.iter().take(threshold).map(|v| gen.exp(v)).collect();

        let mut com_futs = FuturesUnordered::new();

        let comsid = sid.derive_ssid(FuncId::Fecdsa);
        let comsid2 = comsid.next();

        let mut data = vec![0u8; threshold*E::BYTES + E::Ford::BYTES];
        data.chunks_exact_mut(E::BYTES)
            .zip(point_coms.iter())
            .for_each(|(c, p)| {
                p.to_bytes(c);
            });

        for (idx, p) in parties.iter().enumerate().filter(|(_, p)| **p != self.party_id) {
            com_futs.push({
                let p = *p;
                // start with the point coms that are common to everyone
                // don't have a dedicated broadcast com
                let mut data = data.clone();
                // and add the individual share for the last party
                shares.vals[idx+1].to_bytes(&mut data[threshold*E::BYTES..]);
                
                let [comsid, comsid2] = if self.party_id < p { [comsid, comsid2] } else { [comsid2, comsid] };
                async move {
                    let comf = self.com.commit_to(comsid, p, data);
                    let expf = self.com.expect_from(comsid2, p);
                    tokio::join!(comf, expf)
                }
            });
        }

        while let Some((r1, r2)) = com_futs.next().await {
            r1?;
            r2?;
        }

        let mut decom_futs = FuturesUnordered::new();

        for (idx, p) in parties.iter().enumerate().filter(|(_, p)| **p != self.party_id) {
            decom_futs.push({
                let p = *p;
                // start with the point coms that are common to everyone
                // don't have a dedicated broadcast com
                let mut data = data.clone();
                // and add the individual share for the last party
                shares.vals[idx+1].to_bytes(&mut data[threshold*E::BYTES..]);
                let len = data.len();
                
                let [comsid, comsid2] = if self.party_id < p { [comsid, comsid2] } else { [comsid2, comsid] };
                async move {
                    let comf = self.com.decommit_to(comsid, p, data);
                    let expf = async {
                        self.com.value_from(comsid2, p, len).await
                            .map(|bs| {
                                let share_j = E::Ford::from_bytes(&bs[threshold*E::BYTES..]);
                                let coms_j: Vec<E> = bs.chunks_exact(E::BYTES)
                                    .map(|b| E::from_bytes(b))
                                    .collect();
                                (share_j, coms_j)
                            })
                    };
                    (p, tokio::join!(comf, expf))
                }
            });
        }

        let mut p_vals = point_coms;
        let mut pi = shares.vals[my_idx].clone();

        while let Some((_p, (r1, r2))) = decom_futs.next().await {
            r1?;
            let (pi_j, ps_j) = r2?;
            pi += pi_j;
            p_vals.iter_mut().zip(ps_j.into_iter()).for_each(|(old, new)| {
                *old += new;
            });
        }

        let com_pi = gen.exp(&pi);

        let p0 = p_vals[0].clone();


        let cheat = if my_idx < threshold {
            com_pi != p_vals[my_idx]
        } else {
            let mut p_points = vec![points[my_idx].clone()];
            p_points.extend_from_slice(&points[1..threshold]);
            p_vals[0] = com_pi.clone();

            // custom impl since the poly one assumes ring instead of ExpGroup
            let test_zero = p_points
                .iter()
                .zip(p_vals.iter())
                .map(|(xi, yi)| {
                    let l = lagrange_poly(&p_points, xi, |x| -x.clone());
                    return yi.exp(&l);
                })
                .sum();

            p0 != test_zero
        };

        let msg = [cheat as u8];

        let netsid = sid.derive_ssid(FuncId::Fecdsa);

        let res= broadcast_opportunistic(FuncId::Fecdsa, netsid, msg, self.party_id, parties, self.net.clone()).await?;


        for (k,v) in res.iter() {
            if v[0] == 1 {
                return Err(self.cheat(sid, None, format!("Received abort from {}", *k)).into());
            }
        }

        let share = KeyShare {
            pk: p0,
            sk_share: pi,
            party_id: self.party_id,
            all_parties: parties.to_vec(),
            threshold: threshold,
        };

        Ok(share)
    }

}

pub mod internal {
    use super::*;
    pub fn gen_dl_keyshares<E: EcGroup>(parties: &[PartyId], threshold: usize) -> Vec<KeyShare<E>> 
        where E::Ford: RandElement
    {
        let mut points = vec![E::Ford::zero()];
        points.extend(parties.iter().map(|p| E::Ford::from(*p as u64)));

        let mut rng = rand::thread_rng();
        let shares = InterpolationPolynomial::rand_share(&mut rng, threshold, &points);

        let sk = shares.vals[0].clone();
        let gen = E::one();
        let pk = gen.exp(&sk);

        shares.vals
            .into_iter()
            .skip(1)
            .zip(parties.iter())
            .map(|(v, p)| {
                KeyShare {
                    pk: pk.clone(),
                    sk_share: v,
                    party_id: *p,
                    all_parties: parties.to_vec(),
                    threshold
                }
            }).collect()
    }

}

impl<FN, FV, FC, FZ> Dkls23EcdsaPlayer<FN, FV, FC, FZ> {
    pub fn new(party: PartyId, net: Arc<FN>, vole: Arc<FV>, com: Arc<FC>, zero: Arc<FZ>) -> Self {
        Self {
            party_id: party,
            net,
            vole,
            com,
            zero: zero
        }
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;

    use crate::{
        func_net::tests::{get_test_party_infos, build_test_nets},
        func_eot::zzzr23_eot::tests::build_test_eots,
        func_cote::softspoken_ote::tests::build_test_ots,
        func_vole::dkls23_vole::tests::build_test_voles,
        func_com::tests::build_test_coms,
        func_zero::folklore_ro_zero::tests::build_test_zeros,
        //ff2_128::FF2_128,
        field::ExpGroup,
        p256::{P256, P256Scalar},
    };
    use std::sync::Arc;
    use tokio::task::{JoinSet, JoinHandle};

    pub fn build_test_ecdsas<FN, FV, FC, FZ>(
        nets: &[Arc<FN>],
        voles: &[Arc<FV>],
        coms: &[Arc<FC>],
        zeros: &[Arc<FZ>],
    ) -> Vec<Arc<Dkls23EcdsaPlayer<FN, FV, FC, FZ>>> {
        let num = nets.len() as PartyId;
        (1..=num)
            .map(|i| {
                let idx = (i-1) as usize;
                Arc::new(Dkls23EcdsaPlayer::new(i, nets[idx].clone(), voles[idx].clone(), coms[idx].clone(), zeros[idx].clone()))
            })
            .collect()
    }

    #[test]
    fn test_gen_dl_keyshares() {
        let shares = internal::gen_dl_keyshares::<P256>(&[1,2], 2);
        let pk = shares[0].pk.clone();

        let sk_p = shares.iter().map(|s| s.additive_share(&[1,2])).sum();

        let gen = P256::one();
        assert_eq!(pk, gen.exp(&sk_p));
    }

    async fn run_ecdsa_setup() -> Vec<Result<KeyShare<P256>, CheatOrUnexpectedError>> {
        let party_info = get_test_party_infos(3);
        let funcs = vec![FuncId::Fecdsa, FuncId::Fcom, FuncId::Fvole, FuncId::Fot, FuncId::Feot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);
        let voles = build_test_voles(&nets, &ots);
        let coms = build_test_coms(&nets);
        let zeros = build_test_zeros(&coms);
        let ecdsas = build_test_ecdsas(&nets, &voles, &coms, &zeros);

        let sid = SessionId::new(FuncId::Ftest);

        let mut set = JoinSet::new();

        let parties = [1,2,3];

        ecdsas.into_iter()
            .for_each(|ecdsa| {
                set.spawn(async move {
                    let now = std::time::Instant::now();
                    let res = ecdsa.setup(sid, &parties, 2).await;
                    log::info!("Setup took {}ms", now.elapsed().as_millis());
                    res
                });
            });

        set.join_all().await
    }

    #[test]
    fn test_ecdsa_setup() {
        let rt = tokio::runtime::Builder::new_multi_thread()
                            .enable_all()
                            .thread_stack_size(1 << 23)
                            .build()
                            .expect("couldn't build runtime");

        rt.block_on(Box::pin(test_ecdsa_setup_inner()));
    }

    //#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_ecdsa_setup_inner() {

        //todo!("Compiler ICE when trying to compile const generics");
        let _ = env_logger::builder().is_test(true).try_init();


        let results = run_ecdsa_setup().await;

        let mut pk: Option<P256> = None;
        let mut sk = P256Scalar::zero();

        for r in results {
            let share = r.expect("failed to run setup");
            sk += share.additive_share(&[1,2,3]);

            if pk.is_some() {
                assert_eq!(pk.as_ref().unwrap(), &share.pk);
            } else {
                pk = Some(share.pk)
            }
        }

        let gen = P256::one();
        assert_eq!(pk.unwrap(), gen.exp(&sk));
    }


    async fn run_ecdsa_sign() -> (Result<EcdsaSignature<P256>, CheatOrUnexpectedError>, Result<EcdsaSignature<P256>, CheatOrUnexpectedError>) {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Fecdsa, FuncId::Fcom, FuncId::Fvole, FuncId::Fot, FuncId::Feot];
        let nets = build_test_nets(&party_info, funcs).await;
        let eots = build_test_eots(&nets);
        let ots = build_test_ots(&eots, &nets);
        let voles = build_test_voles(&nets, &ots);
        let coms = build_test_coms(&nets);
        let zeros = build_test_zeros(&coms);
        let ecdsas = build_test_ecdsas(&nets, &voles, &coms, &zeros);

        let sid = SessionId::new(FuncId::Ftest);

        let (share0, share1) = {
            let shares = internal::gen_dl_keyshares(&[1,2], 2);
            (shares[0].clone(), shares[1].clone())
        };

        let msg = "Test signature please ignore";

        let t1: JoinHandle<Result<_, _>> = tokio::spawn({
            let ecdsa = ecdsas[0].clone();
            let share = share0;
            async move {
                let now = std::time::Instant::now();
                let res = ecdsa.sign(sid, &[1,2], 0, share, msg.as_bytes()).await;
                log::info!("Signature took {}ms", now.elapsed().as_millis());
                res
            }
        });

        let t2: JoinHandle<Result<_, _>> = tokio::spawn({
            let ecdsa = ecdsas[1].clone();
            let share = share1;
            async move {
                ecdsa.sign(sid, &[1,2], 0, share, msg.as_bytes()).await
            }
        });

        tokio::try_join!(t1, t2).expect("Error running signatures")
    }

    #[test]
    fn test_ecdsa_sign() {
        let rt = tokio::runtime::Builder::new_multi_thread()
                            .enable_all()
                            .thread_stack_size(1 << 23)
                            .build()
                            .expect("couldn't build runtime");

        rt.block_on(Box::pin(test_ecdsa_sign_inner()));
    }

    //#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_ecdsa_sign_inner() {

        //todo!("Compiler ICE when trying to compile const generics");
        let _ = env_logger::builder().is_test(true).try_init();


        match run_ecdsa_sign().await {
            (Ok(sig), Ok(sig2)) => {
                assert_eq!(sig, sig2);
            },
            (Err(e), _) | (_, Err(e)) => { assert!(false, "{:?}", e); }
        };
    }

}
