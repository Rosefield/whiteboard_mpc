use crate::{
    auth_bits::{Abits, ThreshAbits},
    base_func::{BaseFunc, CheatOrUnexpectedError, FuncId, SessionId, UnexpectedError},
    circuits::{CircuitCollection, CircuitElement},
    common_protos::{broadcast_commit_open, open_abits, random_shares},
    field::{Field, RandElement},
    func_abit::AsyncAbit,
    func_com::AsyncCom,
    func_mult::AsyncMult,
    func_net::AsyncNet,
    func_rand::AsyncRand,
    party::PartyId,
    polynomial::lagrange_poly,
};

use anyhow::Context;
use log::trace;

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

#[derive(Debug)]
pub struct RstTabitPlayer<T, FA, FR, FM, FC, FN> {
    party_id: PartyId,
    n: usize,
    t: usize,
    abit: Arc<FA>,
    rand: Arc<FR>,
    mult: Arc<FM>,
    com: Arc<FC>,
    net: Arc<FN>,
    deltas: RwLock<HashMap<SessionId, T>>,
}

impl<T, FA, FR, FM, FC, FN> BaseFunc for RstTabitPlayer<T, FA, FR, FM, FC, FN> {
    const FUNC_ID: FuncId = FuncId::Ftabit;
    const REQUIRED_FUNCS: &'static [FuncId] = &[
        FuncId::Fabit,
        FuncId::Frand,
        FuncId::Fmult,
        FuncId::Fcom,
        FuncId::Fnet,
    ];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// Trait to represent threshold authenticated bit generation
pub trait AsyncTabit<T> {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId, delta: T) -> Result<(), UnexpectedError>;

    /// Sample uniform threshold abits
    async fn sample(
        &self,
        sid: SessionId,
        num_bits: usize,
    ) -> Result<ThreshAbits<T>, CheatOrUnexpectedError>;

    /// Reshare existing abits as thresh abits
    async fn reshare(
        &self,
        sid: SessionId,
        abits: &Abits<T>,
    ) -> Result<ThreshAbits<T>, CheatOrUnexpectedError>;
}

impl<
        T: Field + CircuitElement + RandElement + Copy,
        FA: AsyncAbit<T>,
        FR: AsyncRand,
        FM: AsyncMult<T>,
        FC: AsyncCom,
        FN: AsyncNet,
    > AsyncTabit<T> for RstTabitPlayer<T, FA, FR, FM, FC, FN>
{
    async fn init(&self, sid: SessionId, delta: T) -> Result<(), UnexpectedError> {
        {
            let mut ds = self.deltas.write().unwrap();
            ds.insert(sid, delta);
        }

        let ssid = sid.derive_ssid(FuncId::Ftabit);

        trace!("{}: init ({sid}) abit", self.party_id);
        self.abit
            .init(ssid, delta)
            .await
            .with_context(|| self.err(sid, "Failed to initialize Fabit"))?;
        self.mult
            .init(ssid)
            .await
            .with_context(|| self.err(sid, "Failed to initialize Fmult"))?;
        self.rand
            .init(ssid)
            .await
            .with_context(|| self.err(sid, "Failed to initialize Frand"))?;

        Ok(())
    }

    async fn sample(
        &self,
        sid: SessionId,
        num_bits: usize,
    ) -> Result<ThreshAbits<T>, CheatOrUnexpectedError> {
        // sample random secret shares
        let num_els = (num_bits + T::BIT_SIZE - 1) / T::BIT_SIZE;

        let delta = { self.deltas.read().unwrap()[&sid].clone() };

        let parties: Vec<PartyId> = (1..(self.n + 1) as PartyId).collect();
        let party_points: Vec<_> = (1..self.n + 1).map(|p| T::from(p as u64)).collect();

        let my_shares = random_shares::<T, FN>(
            num_els,
            self.party_id,
            &parties,
            self.t.into(),
            FuncId::Ftabit,
            sid, 
            self.net.clone(),
        )
        .await
        .with_context(|| self.err(sid, "Failed to create initial random shares [[r]]"))?;

        // authenticate the bits of all of the shares
        let mut bits = vec![false; num_els * T::BIT_SIZE];
        my_shares.to_bits(&mut bits);

        let ssid = sid.derive_ssid(FuncId::Ftabit);

        let abits = self
            .abit
            .abit(ssid, bits)
            .await
            .with_context(|| self.err(sid, "Failed to authenticate bit decomp of [[r]]"))?;

        // run check
        // need num_els + n + 1 shared field elements
        let rand_bytes = self
            .rand
            .rand(ssid, (num_els + self.n + 1) * T::BYTES)
            .await?;
        let els: Vec<_> = rand_bytes
            .chunks_exact(T::BYTES)
            .map(|c| T::from_bytes(c))
            .collect();

        let chis = &els[..num_els];
        let cs = &els[num_els..num_els + self.n];
        let r = els.last().unwrap();

        // Split into two sets, one of size t, and the
        let s1 = &party_points[..self.t];
        let s2_idx = std::cmp::min(self.t, self.n - self.t);
        let s2 = &party_points[s2_idx..];

        let gen = T::gen();
        let mut pows = Vec::with_capacity(T::BIT_SIZE);
        pows.push(T::one());
        for i in 1..T::BIT_SIZE {
            pows.push(gen * pows[i - 1]);
        }

        // row el, col party
        // calculate \macof{j}{\State_a^i}, \keyof{i}{\State_a^j}
        // where mijs[j *num_els + a] = \macof{j}{\State_a^i}
        let mut mijs = vec![T::zero(); (self.n - 1) * num_els];
        let mut kijs = vec![T::zero(); (self.n - 1) * num_els];

        for j in 0..self.n - 1 {
            abits.macs[j]
                .chunks_exact(T::BIT_SIZE)
                .zip(mijs[j * num_els..(j + 1) * num_els].iter_mut())
                .for_each(|(ms, mij)| *mij = ms.iter().zip(pows.iter()).map(|(m, p)| *m * p).sum());

            abits.keys[j]
                .chunks_exact(T::BIT_SIZE)
                .zip(kijs[j * num_els..(j + 1) * num_els].iter_mut())
                .for_each(|(ks, kij)| *kij = ks.iter().zip(pows.iter()).map(|(k, p)| *k * p).sum());
        }

        // calculate the lagrange coefficients
        // for each party in the two sets of parties
        let lps1: Vec<T> = s1
            .iter()
            .map(|i| lagrange_poly(s1, i, |xj| *r - xj))
            .collect();

        let lps2: Vec<T> = s2
            .iter()
            .map(|i| lagrange_poly(s2, i, |xj| *r - xj))
            .collect();

        let f = |s: &T, i: usize| {
            let mut acc = T::zero();
            if i <= self.t {
                acc += lps1[i - 1] * s;
            }
            if i > s2_idx {
                acc += lps2[i - s2_idx - 1] * s;
            }

            acc
        };

        // apply shamir reconstruction to the abits
        // \ptabit Check Step 2.
        let yias: Vec<_> = my_shares
            .iter()
            .map(|s| f(s, self.party_id.into()))
            .collect();

        // for the macs we always interpolate for our id
        mijs.iter_mut().for_each(|m| {
            *m = f(m, self.party_id.into());
        });

        let other_ids: Vec<_> = parties
            .iter()
            .filter(|&p| *p != self.party_id)
            .copied()
            .collect();

        // for keys we interpolate for the other party
        // this way we calculate f(M_j^i(y), i) = F(K_j(y), i) + y Delta_j
        kijs.chunks_exact_mut(num_els)
            .zip(other_ids.iter())
            .for_each(|(ks, &j)| {
                ks.iter_mut().for_each(|k| {
                    *k = f(k, j.into());
                });
            });

        // start (n \times l) matrix
        let zias = kijs
            .chunks_exact_mut(num_els)
            // sum over y axis to combine parties
            .reduce(|ls, rs| {
                ls.iter_mut().zip(rs.iter()).for_each(|(l, r)| *l += r);
                ls
            })
            .unwrap();
        // add expected correlation for each element
        zias.iter_mut()
            .zip(yias.into_iter())
            .for_each(|(z, y)| *z += y * delta);

        // reduce expected correlations for all elements to single value
        let zi: T = zias
            .iter()
            .zip(chis.iter())
            .map(|(z, x)| z.clone() * x)
            .sum();

        // start (n \times l) matrix
        let mis: Vec<T> = mijs
            // reduce over x axis to combine elements
            .chunks_exact(num_els)
            .map(|ms| ms.iter().zip(chis.iter()).map(|(m, x)| *m * x).sum())
            .collect();

        let mut m: T = cs[(self.party_id - 1) as usize] * zi;
        // reduce over all parties
        m += other_ids
            .iter()
            .zip(mis.iter())
            .map(|(&i, mi)| cs[(i - 1) as usize] * mi)
            .sum::<T>();

        // Run a multiplication with a random element
        // to test that \sum_i m_i is zero
        let r = {
            let mut rng = rand::thread_rng();
            T::rand(&mut rng)
        };
        let tshare = self
            .mult
            .mult(ssid, r, m)
            .await
            .with_context(|| self.err(sid, "Failed to perform the zero-test multiplication"))?;

        let mut t_bytes = vec![0; T::BYTES];
        tshare.to_bytes(&mut t_bytes);

        let ssid = sid.derive_ssid(FuncId::Ftabit);

        let others =
            broadcast_commit_open(ssid, &t_bytes, self.party_id, &parties, self.com.clone())
                .await
                .with_context(|| self.err(sid, "Failed to open the secret shares of [z]"))?;

        let test: T = tshare + others.iter().map(|o| T::from_bytes(&o)).sum::<T>();

        if !test.is_zero() {
            return Err(self.cheat(sid, None, "Authenticated bits do not represent a degree t-1 polynomial or adversary attempted to spoof a MAC".into()).into());
        }

        let tabits = ThreshAbits::from_abits(num_bits, my_shares, abits);

        Ok(tabits)
    }

    async fn reshare(
        &self,
        sid: SessionId,
        abits: &Abits<T>,
    ) -> Result<ThreshAbits<T>, CheatOrUnexpectedError> {
        // resharing abits <x>
        let delta = { self.deltas.read().unwrap()[&sid].clone() };
        let nbits = abits.len();
        // sample random thresh abits <<r>>
        let rs = self.sample(sid, nbits).await?;

        // convert all thresh to abits
        let all_bits: Vec<usize> = (0..nbits).collect();

        let parties: Vec<PartyId> = (1..(self.n + 1) as PartyId).collect();
        let party_points: Vec<_> = (1..self.n + 1).map(|p| T::from(p as u64)).collect();
        let my_point = party_points[(self.party_id - 1) as usize];

        // convert <<r>> -> <r>
        let mut r_a = rs.convert(&my_point, &party_points, &party_points, &all_bits);
        // calculate and open masked (x+r) = <x> + <r>
        r_a.add_assign(abits);

        let cs = open_abits(
            &r_a,
            self.net.clone(),
            &delta,
            self.party_id.into(),
            &parties,
            sid,
            FuncId::Ftabit,
        )
        .await
        .with_context(|| self.err(sid, "Failed to open abits"))?;

        // create threshold shares using the masked values, <<x>> = (x+r) + <<r>>
        let mut tabits = rs;
        tabits.add_consts(&cs, delta);

        Ok(tabits)
    }
}

impl<T: Field, FA, FR, FM, FC, FN> RstTabitPlayer<T, FA, FR, FM, FC, FN> {
    pub fn new(
        party_id: PartyId,
        n: PartyId,
        t: u16,
        abit: Arc<FA>,
        rand: Arc<FR>,
        mult: Arc<FM>,
        com: Arc<FC>,
        net: Arc<FN>,
    ) -> Result<Self, ()> {
        Ok(RstTabitPlayer {
            party_id: party_id,
            n: n.into(),
            t: t.into(),
            abit: abit,
            rand: rand,
            mult: mult,
            com: com,
            net: net,
            deltas: RwLock::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        ff2_128::FF2_128,
        func_abit::tests::build_test_abits,
        func_com::tests::build_test_coms,
        func_cote::kos_cote::tests::build_test_cotes,
        func_mult::tests::build_test_mults,
        func_net::tests::{build_test_nets, get_test_party_infos},
        func_rand::tests::build_test_rands,
    };

    use rand::Rng;
    use tokio::task::JoinSet;

    pub fn build_test_tabits<
        FA: AsyncAbit<FF2_128>,
        FR: AsyncRand,
        FM: AsyncMult<FF2_128>,
        FC: AsyncCom,
        FN: AsyncNet,
    >(
        abits: &[Arc<FA>],
        rands: &[Arc<FR>],
        mults: &[Arc<FM>],
        coms: &[Arc<FC>],
        nets: &[Arc<FN>],
    ) -> Vec<Arc<RstTabitPlayer<FF2_128, FA, FR, FM, FC, FN>>> {
        let num = abits.len();
        (1..=num)
            .map(|i| {
                Arc::new(
                    RstTabitPlayer::new(
                        i as PartyId,
                        num as PartyId,
                        (num - 1) as PartyId,
                        abits[i - 1].clone(),
                        rands[i - 1].clone(),
                        mults[i - 1].clone(),
                        coms[i - 1].clone(),
                        nets[i - 1].clone(),
                    )
                    .unwrap(),
                )
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_sample_tabits() -> Result<(), ()> {
        let n = 3;
        let party_info = get_test_party_infos(n as PartyId);

        let nets = build_test_nets(
            &party_info,
            vec![FuncId::Fcom, FuncId::Fcote, FuncId::Fmult, FuncId::Ftabit],
        )
        .await;
        let abits = build_test_abits(&party_info);
        let coms = build_test_coms(&nets);
        let rands = build_test_rands(&coms);
        let cotes = build_test_cotes(&nets, &party_info);
        let mults = build_test_mults(&nets, &cotes);
        let tabits = build_test_tabits(&abits, &rands, &mults, &coms, &nets);

        let mut js = JoinSet::<Result<_, CheatOrUnexpectedError>>::new();
        // deliberately not divisible by the field size
        let nbits = 200;
        for (i, tabit) in tabits.into_iter().enumerate() {
            js.spawn(async move {
                let delta = {
                    let mut rng = rand::thread_rng();
                    FF2_128::rand(&mut rng)
                };
                let sid = SessionId::new(FuncId::Ftest);
                let _ = tabit.init(sid, delta).await?;
                let tabits = tabit.sample(sid, nbits).await?;

                assert!(tabits.nbits == nbits);

                let all_points: Vec<_> = (1..=3).map(|i| FF2_128::from(i)).collect();
                let my_point = all_points[i];

                let bit_idx: Vec<usize> = (0..nbits).collect();
                // convert to the 3 sets of pairs

                let mut s1 = [my_point, all_points[(i + 1) % 3]];
                if i == 2 {
                    s1.swap(0, 1);
                }
                let mut s2 = [my_point, all_points[(i + 2) % 3]];
                if i > 0 {
                    s2.swap(0, 1);
                }

                let s1_abits = tabits.convert(&my_point, &all_points, &s1, &bit_idx);
                let s2_abits = tabits.convert(&my_point, &all_points, &s2, &bit_idx);

                Ok((i + 1, delta, s1_abits, s2_abits))
            });
        }

        let mut res: HashMap<usize, _> = HashMap::new();
        while let Some(x) = js.join_next().await {
            let (i, d, s1, s2) = x.unwrap().unwrap();
            res.insert(i, (d, s1, s2));
        }

        let d1 = res[&1].0;
        let d2 = res[&2].0;
        let d3 = res[&3].0;

        for k in 0..nbits {
            let b11 = res[&1].1.bits[k];
            let b12 = res[&1].2.bits[k];

            let b21 = res[&2].1.bits[k];
            let b22 = res[&2].2.bits[k];

            let b31 = res[&3].1.bits[k];
            let b32 = res[&3].2.bits[k];

            // each subset should get the same bit
            assert_eq!(b11 ^ b22, b12 ^ b31);
            assert_eq!(b21 ^ b32, b12 ^ b31);

            let c = |a1: &Abits<_>, a2: &Abits<_>, d2| {
                let mut emac = a2.keys[0][k];
                if a1.bits[k] {
                    emac += d2;
                }
                emac == a1.macs[0][k]
            };

            // all pairwise macs should be valid.
            assert!(c(&res[&1].1, &res[&2].2, d2));
            assert!(c(&res[&2].2, &res[&1].1, d1));

            assert!(c(&res[&1].2, &res[&3].1, d3));
            assert!(c(&res[&3].1, &res[&1].2, d1));

            assert!(c(&res[&2].1, &res[&3].2, d3));
            assert!(c(&res[&3].2, &res[&2].1, d2));
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_reshare_tabits() -> Result<(), ()> {
        let n = 3;
        let party_info = get_test_party_infos(n as PartyId);

        let nets = build_test_nets(
            &party_info,
            vec![FuncId::Fcom, FuncId::Fcote, FuncId::Fmult, FuncId::Ftabit],
        )
        .await;
        let abits = build_test_abits(&party_info);
        let coms = build_test_coms(&nets);
        let rands = build_test_rands(&coms);
        let cotes = build_test_cotes(&nets, &party_info);
        let mults = build_test_mults(&nets, &cotes);
        let tabits = build_test_tabits(&abits, &rands, &mults, &coms, &nets);

        // deliberately not divisible by the field size
        let nbits = 200;

        let deltas: Vec<_> = {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| FF2_128::rand(&mut rng)).collect()
        };

        let mut ebits = vec![false; nbits];

        let abits = {
            let mut rng = rand::thread_rng();
            let bits: Vec<_> = (0..n)
                .map(|_| {
                    let mut bs = vec![false; nbits];
                    rng.fill(&mut bs[..]);
                    bs
                })
                .collect();

            bits.iter().for_each(|bs| {
                ebits.iter_mut().zip(bs.iter()).for_each(|(e, &b)| *e ^= b);
            });

            let mut abits: Vec<_> = (0..n)
                .map(|i| Abits {
                    bits: bits[i].clone(),
                    macs: Vec::new(),
                    keys: (0..n - 1)
                        .map(|_| (0..nbits).map(|_| FF2_128::rand(&mut rng)).collect())
                        .collect(),
                })
                .collect();

            for i in 0..n {
                abits[i].macs = (0..n)
                    .filter(|&x| x != i)
                    .map(|j| {
                        let ji = if i < j { i } else { i - 1 };
                        abits[j].keys[ji]
                            .iter()
                            .zip(abits[i].bits.iter())
                            .map(|(k, &b)| {
                                let mut m = k.clone();
                                if b {
                                    m += deltas[j];
                                }
                                m
                            })
                            .collect()
                    })
                    .collect();
            }
            abits
        };

        let mut js = JoinSet::new();

        for (i, ((tabit, delta), abit)) in tabits
            .into_iter()
            .zip(deltas.into_iter())
            .zip(abits.into_iter())
            .enumerate()
        {
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                let _ = tabit.init(sid, delta).await?;

                let abit = abit;
                let tabits = tabit.reshare(sid, &abit).await?;

                assert!(tabits.nbits == nbits);

                let all_points: Vec<_> = (1..=3).map(|i| FF2_128::from(i)).collect();
                let my_point = all_points[i];

                let bit_idx: Vec<usize> = (0..nbits).collect();
                // convert to the 3 sets of pairs

                let mut s1 = [my_point, all_points[(i + 1) % 3]];
                if i == 2 {
                    s1.swap(0, 1);
                }
                let mut s2 = [my_point, all_points[(i + 2) % 3]];
                if i > 0 {
                    s2.swap(0, 1);
                }

                let s1_abits = tabits.convert(&my_point, &all_points, &s1, &bit_idx);
                let s2_abits = tabits.convert(&my_point, &all_points, &s2, &bit_idx);

                Ok((i + 1, delta, s1_abits, s2_abits))
            });
        }

        let mut res: HashMap<usize, _> = HashMap::new();
        while let Some(x) = js.join_next().await {
            let r: Result<_, CheatOrUnexpectedError> = x.unwrap();
            let (i, d, s1, s2) = r.unwrap();
            res.insert(i, (d, s1, s2));
        }

        let d1 = res[&1].0;
        let d2 = res[&2].0;
        let d3 = res[&3].0;

        for k in 0..nbits {
            let b11 = res[&1].1.bits[k];
            let b12 = res[&1].2.bits[k];

            let b21 = res[&2].1.bits[k];
            let b22 = res[&2].2.bits[k];

            let b31 = res[&3].1.bits[k];
            let b32 = res[&3].2.bits[k];

            // each subset should get the same bit
            // and it should be the same as the original values
            assert_eq!(b11 ^ b22, b12 ^ b31);
            assert_eq!(b21 ^ b32, b12 ^ b31);
            assert_eq!(b11 ^ b22, ebits[k]);

            let c = |a1: &Abits<_>, a2: &Abits<_>, d2| {
                let mut emac = a2.keys[0][k];
                if a1.bits[k] {
                    emac += d2;
                }
                emac == a1.macs[0][k]
            };

            // all pairwise macs should be valid.
            assert!(c(&res[&1].1, &res[&2].2, d2));
            assert!(c(&res[&2].2, &res[&1].1, d1));

            assert!(c(&res[&1].2, &res[&3].1, d3));
            assert!(c(&res[&3].1, &res[&1].2, d1));

            assert!(c(&res[&2].1, &res[&3].2, d3));
            assert!(c(&res[&3].2, &res[&2].1, d2));
        }

        Ok(())
    }
}
