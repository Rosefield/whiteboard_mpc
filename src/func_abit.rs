use crate::{
    auth_bits::Abits,
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    circuits::CircuitElement,
    ffi::ffi::{
        make_abit_player, make_network, EmpAbit, Network as FFI_Network, PartyInfo as FFI_Party,
    },
    field::Field,
    party::{PartyId, PartyInfo},
};

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
};

use anyhow::Context;
use log::{info, trace};
use rand::Rng;

pub struct WrkAbitPlayer<T> {
    party_id: PartyId,
    n: usize,
    party_info: Vec<PartyInfo>,
    net: Arc<OnceLock<cxx::SharedPtr<FFI_Network>>>,
    //rand: Arc<FR>,
    abits: RwLock<HashMap<SessionId, (T, cxx::SharedPtr<EmpAbit>)>>,
}

impl<T> BaseFunc for WrkAbitPlayer<T> {
    const FUNC_ID: FuncId = FuncId::Fabit;
    const REQUIRED_FUNCS: &'static [FuncId] = &[];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// Trait to represent authenticated bit generation
pub trait AsyncAbit<T> {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId, delta: T) -> Result<(), UnexpectedError>;

    /// Authenticate shares from each party
    async fn abit(&self, sid: SessionId, bits: Vec<bool>) -> Result<Abits<T>, UnexpectedError>;
}

impl<T: Field + CircuitElement + 'static> AsyncAbit<T> for WrkAbitPlayer<T> {
    async fn init(&self, sid: SessionId, delta: T) -> Result<(), UnexpectedError> {
        trace!("{}: init ({sid})", self.party_id);
        let party_info: Vec<_> = self
            .party_info
            .iter()
            .map(|p| FFI_Party {
                id: p.id,
                ip: p.ip.to_string(),
                port: p.port,
            })
            .collect();

        let mut delta_b = vec![false; T::BIT_SIZE];
        delta.to_bits(&mut delta_b);

        let id = self.party_id;
        let net_lock = self.net.clone();
        let abit = tokio::task::spawn_blocking(move || {
            let net = net_lock.get_or_init(|| {
                let n = make_network(id, &party_info, 200).unwrap();

                n
            });
            let parties: Vec<_> = party_info.iter().map(|p| p.id).collect();

            return make_abit_player(id, &parties, net.clone(), &delta_b);
        })
        .await
        .unwrap()
        .with_context(|| self.err(sid, "Failed to create abit player"))?;

        {
            let mut map = self.abits.write().unwrap();
            map.insert(sid, (delta, abit));
        }

        Ok(())
    }

    /// Authenticate shares from each party
    async fn abit(&self, sid: SessionId, bits: Vec<bool>) -> Result<Abits<T>, UnexpectedError> {
        let abit = {
            let map = self.abits.read().unwrap();
            map[&sid].1.clone()
        };

        let n = self.n;

        trace!("{}: sid {} make abits", self.party_id, sid);
        let (nb, t) = tokio::task::spawn_blocking(move || {
            let mut bits = bits;
            let mut rng = rand::thread_rng();
            let nbits = bits.len();
            // emp-tool expects an extra 3*SSP random elements
            let full_len = nbits + 3 * 80;
            bits.resize(full_len, false);
            rng.fill(&mut bits[nbits..]);
            let mut macs = vec![vec![T::zero(); full_len]; n - 1];
            let mut keys = vec![vec![T::zero(); full_len]; n - 1];

            let nbytes = unsafe {
                let mut mrefs: Vec<&mut [T]> = macs.iter_mut().map(|m| m.as_mut_slice()).collect();
                let mut krefs: Vec<&mut [T]> = keys.iter_mut().map(|k| k.as_mut_slice()).collect();
                // Safety: T has alignment at least as strict as [u8; 16]
                let mref =
                    std::mem::transmute::<&mut [&mut [T]], &mut [&mut [[u8; 16]]]>(&mut mrefs);
                let kref =
                    std::mem::transmute::<&mut [&mut [T]], &mut [&mut [[u8; 16]]]>(&mut krefs);
                let r = abit.create_abits(&bits, mref, kref);
                if r.is_err() {
                    return Err(r.err().unwrap());
                }
                r.unwrap()
            };

            bits.truncate(nbits);
            for m in macs.iter_mut() {
                m.truncate(nbits);
            }
            for k in keys.iter_mut() {
                k.truncate(nbits);
            }

            Ok((nbytes, Abits { bits, macs, keys }))
        })
        .await
        .unwrap()
        .with_context(|| self.err(sid, "Failed to construct abits"))?;

        info!("{}: sid {} net bytes {}", self.party_id, sid, nb);

        Ok(t)

        // if writing our own
        // use OT to authenticate bits and 2k + s extra random bits
        // open last k+s random bits to check consistency of inputs per party
        // open remaining k random bits to check consistency of deltas
    }
}

impl<T: Field> WrkAbitPlayer<T> {
    pub fn new(party_id: PartyId, n: PartyId, party_info: &[PartyInfo]) -> Result<Self, ()> {
        // underlying implementation only supports 128bit size
        assert_eq!(T::BYTES, 16);
        // We are going to transmute vectors of the field element as vectors of byte arrays
        // and we need to ensure that the alignment is sufficient
        assert!(std::mem::align_of::<T>() >= std::mem::align_of::<[u8; 16]>());

        Ok(WrkAbitPlayer {
            party_id: party_id,
            n: n.into(),
            party_info: party_info.to_vec(),
            net: Arc::new(OnceLock::new()),
            abits: RwLock::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        ff2_128::FF2_128,
        field::{ConstInt, RandElement},
        func_net::tests::{build_test_nets, get_test_party_infos},
    };

    use rand::Rng;
    use std::sync::Arc;
    use tokio::task::JoinSet;

    pub fn build_test_abits<T: Field>(party_info: &[PartyInfo]) -> Vec<Arc<WrkAbitPlayer<T>>> {
        let num = party_info.len() as PartyId;
        (1..=num)
            .map(|i| Arc::new(WrkAbitPlayer::new(i, num, &party_info).unwrap()))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_abit_creation() {
        let parties = get_test_party_infos(3);
        let abits = build_test_abits::<FF2_128>(&parties);

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for (i, abit) in abits.into_iter().enumerate() {
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                let (delta, bits) = {
                    let mut rng = rand::thread_rng();
                    let delta = FF2_128::rand(&mut rng);
                    let mut bits = vec![false; 8];
                    rng.fill(&mut bits[..]);
                    (delta, bits)
                };

                let _ = abit.init(sid, delta).await?;

                let a_s = abit.abit(sid, bits.clone()).await?;
                Ok((i + 1, delta, bits, a_s))
            });
        }

        let mut res: HashMap<usize, _> = HashMap::with_capacity(3);

        while let Some(r) = js.join_next().await {
            let (id, delta, bs, a_s) = r.unwrap().unwrap();
            for (b, a) in bs.iter().zip(a_s.bits.iter()) {
                assert_eq!(*b, *a);
            }
            res.insert(id, (delta, a_s));
        }

        let z = FF2_128::zero();
        for i in 1..=3 {
            for j in i + 1..=3 {
                let (i_d, i_a) = &res[&i];
                let (j_d, j_a) = &res[&j];

                for k in 0..j_a.bits.len() {
                    let e_i = i_a.keys[j - 2][k] + if j_a.bits[k] { i_d } else { &z };
                    let e_j = j_a.keys[i - 1][k] + if i_a.bits[k] { j_d } else { &z };
                    assert_eq!(e_i, j_a.macs[i - 1][k]);
                    assert_eq!(e_j, i_a.macs[j - 2][k]);
                }
            }
        }
    }
}
