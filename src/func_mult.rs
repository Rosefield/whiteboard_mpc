use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    field::{FWrap, RandElement, Ring},
    func_cote::AsyncCote,
    func_net::AsyncNet,
    party::PartyId,
};

use std::{marker::PhantomData, sync::Arc};

use anyhow::Context;

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use futures::stream::FuturesUnordered;
use futures::StreamExt;

use sha2::{Digest, Sha256};

use log::trace;

#[derive(Debug)]
pub struct DklsMultPlayer<T, FN, FC> {
    party_id: PartyId,
    n: usize,
    s: usize,
    net: Arc<FN>,
    cote: Arc<FC>,
    _t: PhantomData<T>,
}

impl<T, FN, FC> BaseFunc for DklsMultPlayer<T, FN, FC> {
    const FUNC_ID: FuncId = FuncId::Fmult;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet, FuncId::Fcote];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// Trait to represent the n-party multiplication functionality
/// over a ring T.
pub trait AsyncMult<T: Ring> {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId) -> Result<(), UnexpectedError>;

    /// calculate $\sum_i a_i * \sum_i b_i$ and give an additive share of the output
    /// to each party.
    async fn mult(&self, sid: SessionId, a: T, b: T) -> Result<T, CheatOrUnexpectedError>;
}

impl<T: Ring + RandElement, FN: AsyncNet, FC: AsyncCote> AsyncMult<T>
    for DklsMultPlayer<T, FN, FC>
{
    async fn init(&self, sid: SessionId) -> Result<(), UnexpectedError> {
        trace!("{}: init ({sid})", self.party_id);
        let ssid = sid.derive_ssid(FuncId::Fmult);
        // setup pairwise COTe instances to use for a Gilboa-style multiplication
        for i in (1..=self.n as PartyId).filter(|&i| i != self.party_id) {
            // lower id is sender
            let is_sender = self.party_id < i;
            self.cote
                .init(ssid, i, is_sender)
                .await
                .with_context(|| self.err(sid, format!("Failed to initialize Fcote with {i}")))?;
        }
        Ok(())
    }

    async fn mult(&self, sid: SessionId, a: T, b: T) -> Result<T, CheatOrUnexpectedError> {
        let ssid = sid.derive_ssid(FuncId::Fmult);
        // Do a pairwise multiplication with each other party
        // to calculate the values a_i b_j , a_j b_i
        let futs: FuturesUnordered<_> = (1..=self.n as PartyId)
            .filter(|&i| i != self.party_id)
            .map(|i| {
                let mut a = a.clone();
                let mut b = b.clone();
                // i will provide (a,b) and we want to provide (b, a)
                if i < self.party_id {
                    std::mem::swap(&mut a, &mut b);
                }
                // Two party multiplication will return additive shares
                // of the multiplication
                self.mul_2p(sid, ssid, i, a, b)
            })
            .collect();

        tokio::pin!(futs);

        // Combine all of the pairwise multiplications to get shares of the full value
        // including our term a_i b_i
        let mut c = a * &b;
        while let Some(r) = futs.next().await {
            let (ab_i, ba_i) = r?;
            c += ab_i;
            c += ba_i;
        }

        Ok(c)
    }
}

impl<T: Ring, FN, FC> DklsMultPlayer<T, FN, FC> {
    pub fn new(
        party_id: PartyId,
        n: usize,
        s: usize,
        net: Arc<FN>,
        cote: Arc<FC>,
    ) -> Result<Self, ()> {
        // Restriction for now because I haven't implemented hashing to more than 32 bytes of output
        assert_eq!(T::BYTES, 16);

        Ok(DklsMultPlayer {
            party_id: party_id,
            n: n,
            s: s,
            net: net,
            cote: cote,
            _t: PhantomData,
        })
    }
}

impl<T: Ring + RandElement, FN: AsyncNet, FC: AsyncCote> DklsMultPlayer<T, FN, FC> {
    // for parties i,j calculate shares of (a_i a_j, b_i b_j)
    // could be easily adapted to take a Vec<T> input and calculate each componentwise mult
    // or in box form
    //  a    --------  b
    // ----> |      | <---
    //[ab]_i | MUL  | [ab]_j
    // <---- |      | --->
    //       --------
    async fn mul_2p(
        &self,
        sid: SessionId,
        ssid: SessionId,
        other: PartyId,
        a: T,
        b: T,
    ) -> Result<(T, T), CheatOrUnexpectedError> {
        // zeta is the size of the encoding of the input in bits
        // the addition 2s bits are to hide Bob's input under selective failure attack
        let zeta = T::BYTES * 8 + 2 * self.s;

        let is_sender = self.party_id < other;

        // Use a seed to generate a shared random vector, allowing
        // Bob to pick it as it is for their security.
        let seed = if is_sender {
            let (seed, size) = self
                .net
                .recv_from_local(other, FuncId::Fmult, sid, [0; 32])
                .await
                .with_context(|| {
                    self.err(sid, format!("Failed to receive gadget seed from {other}"))
                })?;
            assert_eq!(size, 32);
            seed
        } else {
            // Bob chooses the seed
            let mut seed = [0; 32];
            {
                let mut rng = rand::thread_rng();
                rng.fill(&mut seed);
            }
            self.net
                .send_to_local(other, FuncId::Fmult, sid, &seed)
                .await
                .with_context(|| self.err(sid, format!("Failed to send gadget seed to {other}")))?;
            seed
        };

        // zeta public random elements of the field
        let gadget: Vec<T> = {
            let mut rng = ChaCha20Rng::from_seed(seed);
            (0..zeta).map(|_| T::rand(&mut rng)).collect()
        };

        // We use RO calls to generate shared random values in the middle of the protocol
        // Of course the RO does not exist, but we have Sha2.
        // We need to create two outputs so set up our two hash instances
        let mut h1 = Sha256::new();
        let mut h2 = Sha256::new();
        {
            let (i, j) = if is_sender {
                (self.party_id, other)
            } else {
                (other, self.party_id)
            };
            // Use the identities of the two parties and the sid for this protocol to generate
            // a new instance of the "RO".
            let ro_seed = Sha256::new()
                .chain_update(&i.to_le_bytes())
                .chain_update(&j.to_le_bytes())
                .chain_update(&ssid.id.to_le_bytes())
                .finalize();

            h1.update(&ro_seed);
            h1.update(&[1]);

            h2.update(&ro_seed);
            h2.update(&[2]);
        }

        // Function that takes the transcript from the COTe protocol to generate
        let trace_fn = |bytes: &[u8]| {
            h1.update(bytes);
            h2.update(bytes);
        };

        // Consume the two hash instances to generate the elements chi needed
        let hash_chi = |h1: Sha256, h2: Sha256| {
            let h1d = h1.finalize();
            let h2d = h2.finalize();
            let chi_t_1 = T::from_bytes(&h1d[..T::BYTES]);
            let chi_t_2 = T::from_bytes(&h1d[T::BYTES..]);
            let chi_h_1 = T::from_bytes(&h2d[..T::BYTES]);
            let chi_h_2 = T::from_bytes(&h2d[T::BYTES..]);

            [FWrap((chi_t_1, chi_h_1)), FWrap((chi_t_2, chi_h_2))]
        };

        // The protocol first performs a random multiplication using the COTe functionality
        // and then once we have checked the consistency of the random multiplication use it
        // to calculate the actual desired multiplication

        if is_sender {
            // I am the sender
            // we are performing two multiplications using OT
            // where we provide an input a multiple times , and bob provides
            // zeta selection bits, to learn m_i + a b_i
            // We do two such multiplications because we want to calculate a_i b_j and b_i a_j
            let mut alphas = Vec::with_capacity(2 * zeta);
            let (a1, a2) = {
                let mut rng = rand::thread_rng();
                // We do the multiplication with two values as the single input
                // hereby tilde a and hat a so that we can check the correctness of
                // each party's input to the COTe
                let a1 = FWrap::<(T, T)>::rand(&mut rng);
                let a2 = FWrap::<(T, T)>::rand(&mut rng);
                (a1, a2)
            };

            // first zeta elements are a1, followed by zeta copies of a2
            alphas.resize(zeta, a1.clone());
            alphas.resize(2 * zeta, a2.clone());

            // learn za which is m_{1,i} || m_{2,i} for i \in zeta
            let za = self
                .cote
                .send_trace(ssid, other, alphas, trace_fn)
                .await
                .with_context(|| self.err(sid, format!("ote send {other} failed")))?;

            // create the random elements from the transcript
            // where chi_i = (\tilde{chi}_i, \hat{chi}_i)
            let chi = hash_chi(h1, h2);

            // vector of consistency check elements and adjustments, r || u || g
            let mut rug_bytes = vec![0; T::BYTES * (zeta + 4)];

            // First calculate r, which is r_j = \sum_l t-chi_l t-m_{l,j} + h-chi_l h-m_{l,j}
            rug_bytes
                .chunks_exact_mut(T::BYTES)
                .zip(za[..zeta].iter())
                .zip(za[zeta..].iter())
                .for_each(|((r, za_1), za_2)| {
                    let mut x = chi[0].clone() * za_1;
                    x += chi[1].clone() * za_2;
                    let r_j = x.0 .0 + x.0 .1;
                    r_j.to_bytes(r);
                });

            // u_l = t-chi_l t-a_l + h-chi_l h-a_l
            // r,u are used by bob to confirm that alice put in the same value
            // a_l for each of the zeta inputs
            let x1 = chi[0].clone() * &a1;
            let u1 = x1.0 .0 + x1.0 .1;
            let x2 = chi[1].clone() * &a2;
            let u2 = x2.0 .0 + x2.0 .1;

            u1.to_bytes(&mut rug_bytes[T::BYTES * zeta..]);
            u2.to_bytes(&mut rug_bytes[T::BYTES * (zeta + 1)..]);

            // gamma is the adjustment value that uses the random multiplication to mask the intended inputs a,b
            let g1 = a.clone() + &a1.0 .0;
            let g2 = b.clone() + &a2.0 .0;

            g1.to_bytes(&mut rug_bytes[T::BYTES * (zeta + 2)..]);
            g2.to_bytes(&mut rug_bytes[T::BYTES * (zeta + 3)..]);

            let _ = self
                .net
                .send_to_local(other, FuncId::Fmult, sid, rug_bytes)
                .await
                .with_context(|| self.err(sid, format!("Failed to send r,u,gamma_a to {other}")))?;

            // receive bob's adjustment values, (b - t-b)
            let gb_bytes = vec![0; 2 * T::BYTES];
            let (gb_bytes, size) = self
                .net
                .recv_from_local(other, FuncId::Fmult, sid, gb_bytes)
                .await
                .with_context(|| {
                    self.err(sid, format!("Failed to receive gamma_b from {other}"))
                })?;

            assert_eq!(size, 2 * T::BYTES);

            let gb_1 = T::from_bytes(&gb_bytes);
            let gb_2 = T::from_bytes(&gb_bytes[T::BYTES..]);

            // calculate the shares of the output
            // note that \sum_k t-za_k gad_k + \sum_k t-zb_k gad_k = (t-a_i t-b_i)
            // or the randomized product
            // so za_adj_1 = (a (b + t-b)) + [(t-a t-b)]
            let za_adj_1 = a * gb_1
                + gadget
                    .iter()
                    .zip(za[..zeta].iter())
                    .map(|(g, z)| {
                        let mut g = g.clone();
                        g *= &z.0 .0;
                        g
                    })
                    .sum::<T>();

            let za_adj_2 = b * gb_2
                + gadget
                    .iter()
                    .zip(za[zeta..].iter())
                    .map(|(g, z)| {
                        let mut g = g.clone();
                        g *= &z.0 .0;
                        g
                    })
                    .sum::<T>();

            Ok((za_adj_1, za_adj_2))
        } else {
            // I am the receiver
            // Sample 2 zeta random bits as a randomized encoding
            let mut betas: Vec<_> = vec![false; 2 * zeta];
            {
                let mut rng = rand::thread_rng();
                rng.fill(&mut betas[..]);
            }

            // Using the gadget vector calculate the elements t-b using the random bits
            let mut bt1: T = T::zero();
            betas[..zeta].iter().zip(gadget.iter()).for_each(|(&b, g)| {
                if b {
                    bt1 += g
                }
            });
            let mut bt2: T = T::zero();
            betas[zeta..].iter().zip(gadget.iter()).for_each(|(&b, g)| {
                if b {
                    bt2 += g
                }
            });

            // receive share of the randomized product
            let zb: Vec<FWrap<(T, T)>> = self
                .cote
                .recv_trace(ssid, other, betas.clone(), trace_fn)
                .await
                .with_context(|| self.err(sid, format!("ote-recv {other} failed")))?;

            let chi = hash_chi(h1, h2);

            // receive the check message (r, u) and adjustment value gamma_a from alice
            let rug_bytes = vec![0; T::BYTES * (zeta + 4)];
            let (rug_bytes, size) = self
                .net
                .recv_from_local(other, FuncId::Fmult, sid, rug_bytes)
                .await
                .with_context(|| {
                    self.err(sid, format!("Failed to receive r,u, gamma_a from {other}"))
                })?;
            assert_eq!(size, T::BYTES * (zeta + 4));

            let rug_els: Vec<T> = rug_bytes
                .chunks_exact(T::BYTES)
                .map(|b| T::from_bytes(b))
                .collect();

            let r = &rug_els[..zeta];
            let u = &rug_els[zeta..zeta + 2];
            let g = &rug_els[zeta + 2..zeta + 4];

            // check that alice put in the same value in each of the zeta
            // OT extensions
            let check = r
                .iter()
                .zip(zb[..zeta].iter().zip(zb[zeta..].iter()))
                .zip(betas[..zeta].iter().zip(betas[zeta..].iter()))
                .all(|((r, (zb_1, zb_2)), (&b_1, &b_2))| {
                    let mut x = chi[0].clone() * zb_1;
                    x += chi[1].clone() * zb_2;
                    let r_j = x.0 .0 + x.0 .1 + r;

                    let mut bu_j = T::zero();
                    if b_1 {
                        bu_j += &u[0];
                    }
                    if b_2 {
                        bu_j += &u[1];
                    }

                    r_j == bu_j
                });

            if !check {
                return Err(self
                    .cheat(
                        ssid,
                        Some(other),
                        "Alice's Fcote correlation input inconsistent".into(),
                    )
                    .into());
            }

            // Calculate our adjustment values (b + t-b)
            let mut gb_bytes = vec![0; 2 * T::BYTES];
            let gb1 = a + &bt1;
            let gb2 = b + &bt2;
            gb1.to_bytes(&mut gb_bytes[..T::BYTES]);
            gb2.to_bytes(&mut gb_bytes[T::BYTES..]);

            let _ = self
                .net
                .send_to_local(other, FuncId::Fmult, sid, gb_bytes)
                .await
                .with_context(|| self.err(sid, format!("Failed to send gamma_b to {other}")))?;

            // calculate our output shares
            // like alice we have the \sum_k t-zb_k + gad_k
            // but we add (t-b (a + t-a))
            // so that our combined sum is (a b)
            let zb_adj_1 = bt1 * &g[0]
                + gadget
                    .iter()
                    .zip(zb[..zeta].iter())
                    .map(|(g, z)| {
                        let mut g = g.clone();
                        g *= &z.0 .0;
                        g
                    })
                    .sum::<T>();

            let zb_adj_2 = bt2 * &g[1]
                + gadget
                    .iter()
                    .zip(zb[zeta..].iter())
                    .map(|(g, z)| {
                        let mut g = g.clone();
                        g *= &z.0 .0;
                        g
                    })
                    .sum::<T>();

            Ok((zb_adj_1, zb_adj_2))
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        ff2_128::FF2_128,
        field::ConstInt,
        func_cote::kos_cote::tests::build_test_cotes,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };

    use std::sync::Arc;

    use tokio::task::JoinSet;

    pub fn build_test_mults<FN, FC>(
        nets: &[Arc<FN>],
        cotes: &[Arc<FC>],
    ) -> Vec<Arc<DklsMultPlayer<FF2_128, FN, FC>>> {
        let n = nets.len();
        (1..=n)
            .map(|i| {
                Arc::new(
                    DklsMultPlayer::new(
                        i as PartyId,
                        n,
                        80,
                        nets[i - 1].clone(),
                        cotes[i - 1].clone(),
                    )
                    .unwrap(),
                )
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_mult() {
        let parties = get_test_party_infos(3);
        let nets = build_test_nets(&parties, vec![FuncId::Fcote, FuncId::Fmult]).await;
        let cotes = build_test_cotes(&nets, &parties);
        let mults = build_test_mults(&nets, &cotes);

        let mut js = JoinSet::<Result<_, CheatOrUnexpectedError>>::new();
        for (i, mult) in mults.into_iter().enumerate() {
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                mult.init(sid).await?;
                let a = FF2_128::new(0, 1 << i);
                let b = FF2_128::new(1 << i, 0);
                mult.mult(sid, a, b).await
            });
        }

        let mut acc = FF2_128::zero();

        while let Some(r) = js.join_next().await {
            let share = r.unwrap().unwrap();
            acc += share;
        }

        assert_eq!(acc, FF2_128::new(0, 7) * FF2_128::new(7, 0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_mult_is_zero() {
        let n = 3;
        let parties = get_test_party_infos(n);
        let nets = build_test_nets(&parties, vec![FuncId::Fcote, FuncId::Fmult]).await;
        let cotes = build_test_cotes(&nets, &parties);
        let mults = build_test_mults(&nets, &cotes);

        // create shares of zero; the first n-1 parties get random and the last
        // party gets the sum of the remaining shares
        let rs: Vec<_> = (1..=n)
            .map(|_| {
                let mut rng = rand::thread_rng();
                FF2_128::rand(&mut rng)
            })
            .collect();
        let mut zs: Vec<_> = (1..n)
            .map(|_| {
                let mut rng = rand::thread_rng();
                FF2_128::rand(&mut rng)
            })
            .collect();

        zs.push(zs.iter().sum());

        let mut js = JoinSet::<Result<_, CheatOrUnexpectedError>>::new();
        for (i, mult) in mults.into_iter().enumerate() {
            let a = rs[i].clone();
            let b = zs[i].clone();
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                mult.init(sid).await?;
                mult.mult(sid, a, b).await
            });
        }

        let mut acc = FF2_128::zero();

        while let Some(r) = js.join_next().await {
            let share = r.unwrap().unwrap();
            acc += share;
        }

        assert_eq!(acc, FF2_128::zero());
    }
}
