use crate::{
    func_cote::AsyncCote,
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    ffi::ffi::{
        make_network, make_ot_player, IknpOte, Network as FFI_Network, PartyInfo as FFI_Party,
    },
    field::Ring,
    func_net::AsyncNet,
    party::{PartyId, PartyInfo},
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::Context;
use log::{info, trace};
use rand::Rng;
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub struct KosCotePlayer<FN> {
    party_id: PartyId,
    party_info: Vec<PartyInfo>,
    net: Arc<FN>,
    ffi_net: Arc<OnceLock<cxx::SharedPtr<FFI_Network>>>,
    otes: Mutex<HashMap<(SessionId, PartyId), ([u8; 16], cxx::SharedPtr<IknpOte>)>>,
}

impl<FN> BaseFunc for KosCotePlayer<FN> {
    const FUNC_ID: FuncId = FuncId::Fcote;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet];

    fn party(&self) -> PartyId {
        self.party_id
    }
}
//
impl<FN: AsyncNet> AsyncCote for KosCotePlayer<FN> {
    async fn init(
        &self,
        sid: SessionId,
        other: PartyId,
        is_sender: bool,
    ) -> Result<(), UnexpectedError> {
        assert!(self.party_id != other);

        let my_id = self.party_id;
        let net_lock = self.ffi_net.clone();

        trace!("{}: init ({sid}) with {other}", self.party_id);
        // Create the OT instance and run the base OT preprocessing
        // The sender provides its correlation delta to select 1 of 2 seeds
        // and thus learns S_delta, and the receiver has the seeds S_0, S_1
        let (delta, ot) = tokio::task::spawn_blocking(move || {
            // delta will be ignored if is_sender is false
            let mut delta = [0u8; 16];
            let mut rng = rand::thread_rng();
            rng.fill(&mut delta);

            // this should not actually run but we may need to
            // wait for init to finish
            let netc = net_lock.get_or_init(|| panic!()).clone();

            make_ot_player(my_id, other, netc, is_sender, delta).map(|ot| (delta, ot))
        })
        .await
        .unwrap()
        .with_context(|| self.err(sid, format!("Failed to create OT with {other}")))?;

        {
            let mut guard = self.otes.lock().unwrap();
            guard.insert((sid, other), (delta, ot));
        }

        Ok(())
    }

    /// As the sender send \vec{alpha}, and receive \vec{omega}, such that the receiver learns \vec{omega} + \vec{beta} * \vec{alpha}
    async fn send<T: Ring>(
        &self,
        sid: SessionId,
        other: PartyId,
        correlations: Vec<T>,
    ) -> Result<Vec<T>, UnexpectedError> {
        self.send_trace(sid, other, correlations, |_| {}).await
    }

    async fn send_trace<T: Ring, F: FnMut(&[u8])>(
        &self,
        sid: SessionId,
        other: PartyId,
        correlations: Vec<T>,
        mut trace_fn: F,
    ) -> Result<Vec<T>, UnexpectedError> {
        // For now since the output of the hash function is being used directly
        // but a larger hash / other means of extending would be fine.
        assert!(T::BYTES <= 32);

        let (delta, ot) = {
            let guard = self.otes.lock().unwrap();
            guard[&(sid, other)].clone()
        };

        let num = correlations.len();
        let ot2 = ot.clone();
        // Run the first step of the IKNP OT extension
        // The receiver calculates V_0, V_1 being the expansion of each row of S_0, S_1
        // and sends u = v_0 + v_1 + b to alice, and outputs V_0
        // the sender receives u and outputs the values Z = V_delta + (delta * U)
        // thus we have a random COT as output
        let block_corr = tokio::task::spawn_blocking(move || {
            let mut block_corr = vec![[0; 16]; num];

            ot.ote_extend_send_rand(&mut block_corr).map(|_| block_corr)
        })
        .await
        .unwrap()
        .with_context(|| {
            self.err(
                sid,
                format!("Failed to do random send extension with {other}"),
            )
        })?;

        info!(
            "{}: sid {} to {} ot bytes {}",
            self.party_id,
            sid,
            other,
            ot2.net_stat()
        );

        // We want to transform our random COT into the correlation we desire
        // This is done by hashing the elements of Z and (Z + delta) to break the correlation
        // and also extend to the appropriate correlation element size.
        // we calculate two values, ta, our output value, and tao the adjustment we send
        // to the receiver.
        // The value H(j|| z_j + delta) is used to mask our desired correlation a_j
        let mut tao_bytes: Vec<u8> = vec![0; T::BYTES * num];
        let ta = block_corr
            .into_iter()
            .zip(tao_bytes.chunks_exact_mut(T::BYTES))
            .zip(correlations.into_iter())
            .enumerate()
            .map(|(i, ((mut z, tao_i), mut a))| {
                // calculate H(j||Z_j), H(j|| Z_j + delta)
                let dig1 = Sha256::new()
                    .chain_update(&(i as u32).to_be_bytes())
                    .chain_update(&z)
                    .finalize();
                let ta_i = T::from_bytes(&dig1);

                z.iter_mut().zip(delta.iter()).for_each(|(b, d)| *b ^= d);

                let dig2 = Sha256::new()
                    .chain_update(&(i as u32).to_be_bytes())
                    .chain_update(&z)
                    .finalize();

                a -= &ta_i;
                a += T::from_bytes(&dig2);

                a.to_bytes(tao_i);

                ta_i
            })
            .collect();

        // Use the last shared message as the transcript/trace of the computation
        trace_fn(&tao_bytes);

        self.net
            .send_to_local(other, FuncId::Fcote, sid, tao_bytes)
            .await
            .with_context(|| self.err(sid, format!("Failed to send tao to {other}")))?;

        Ok(ta)
    }

    /// As the receiver send \vec{beta}, and receive \vec{omega} + \vec{beta} * \vec{alpha}
    async fn recv<T: Ring>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: Vec<bool>,
    ) -> Result<Vec<T>, UnexpectedError> {
        self.recv_trace(sid, other, selections, |_| {}).await
    }

    async fn recv_trace<T: Ring, F: FnMut(&[u8])>(
        &self,
        sid: SessionId,
        other: PartyId,
        selections: Vec<bool>,
        mut trace_fn: F,
    ) -> Result<Vec<T>, UnexpectedError> {
        assert!(T::BYTES <= 32);

        let (_, ot) = {
            let guard = self.otes.lock().unwrap();
            guard[&(sid, other)].clone()
        };

        let ot2 = ot.clone();

        let num = selections.len();
        // Receive the values V_0 that is the expansion of S_0
        let (selections, out_blocks) = tokio::task::spawn_blocking(move || {
            let mut out_blocks = vec![[0; 16]; num];
            let s = selections;

            ot.ote_extend_recv_rand(&s, &mut out_blocks)
                .map(|_| (s, out_blocks))
        })
        .await
        .unwrap()
        .with_context(|| {
            self.err(
                sid,
                format!("Failed to do random recv extension with {other}"),
            )
        })?;

        info!(
            "{}: sid {} to {} ot bytes {}",
            self.party_id,
            sid,
            other,
            ot2.net_stat()
        );

        // Receive the adjustment message from the sender
        let tao_bytes: Vec<u8> = vec![0; T::BYTES * num];
        let (tao_bytes, nbytes) = self
            .net
            .recv_from_local(other, FuncId::Fcote, sid, tao_bytes)
            .await
            .with_context(|| self.err(sid, format!("Failed to receive tao from {other}")))?;
        assert!(nbytes == T::BYTES * num);

        trace_fn(&tao_bytes);

        // If our selection bit is 0, then we just output -H(j|| v_j) as our message
        // which is the same as ta_j as calculated by the sender
        // Otherwise if our bit is 1 use the adjustment tao to create the share of the correlated value.
        let tb = out_blocks
            .into_iter()
            .zip(tao_bytes.chunks_exact(T::BYTES))
            .zip(selections.into_iter())
            .enumerate()
            .map(|(i, ((z, tao_i), b))| {
                let dig1 = Sha256::new()
                    .chain_update(&(i as u32).to_be_bytes())
                    .chain_update(&z)
                    .finalize();
                let mut tb_i = T::from_bytes(&dig1);

                if b {
                    tb_i += T::from_bytes(&tao_i);
                }

                tb_i
            })
            .collect();

        Ok(tb)
    }
}

impl<FN> KosCotePlayer<FN> {
    pub fn new(party_id: PartyId, parties: &[PartyInfo], net: Arc<FN>) -> Result<Self, ()> {
        let ffi_net = Arc::new(OnceLock::new());

        let party_info: Vec<_> = parties
            .iter()
            .map(|p| FFI_Party {
                id: p.id,
                ip: p.ip.to_string(),
                port: p.port,
            })
            .collect();

        let n = ffi_net.clone();

        // TODO: this should probably be improved
        // Spawning the background task is definitely not optimal, but constructing
        // seems to be the most reasonable place where all parties are coordinated at the moment
        // using a std thread instead of tokio runtime so that it doesn't get cancelled
        // when the scope is left
        std::thread::spawn(move || {
            let _ = n.get_or_init(|| make_network(party_id, &party_info, 300).unwrap());
        });

        Ok(KosCotePlayer {
            party_id: party_id,
            party_info: parties.to_vec(),
            net: net,
            ffi_net: ffi_net,
            otes: Mutex::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        ff2_128::FF2_128,
        field::ConstInt,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };

    use tokio::task::JoinSet;

    pub fn build_test_cotes<FN: AsyncNet>(
        nets: &[Arc<FN>],
        party_info: &[PartyInfo],
    ) -> Vec<Arc<KosCotePlayer<FN>>> {
        let num = party_info.len();
        (1..=num)
            .map(|i| {
                Arc::new(KosCotePlayer::new(i as PartyId, party_info, nets[i - 1].clone()).unwrap())
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cote() {
        let parties = get_test_party_infos(3);
        let nets = build_test_nets(&parties, vec![FuncId::Fcote]).await;
        let cotes = build_test_cotes(&nets, &parties);

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for (i, cote) in cotes.into_iter().enumerate() {
            js.spawn(async move {
                // make an instance for each pair
                let sid = SessionId::new(FuncId::Ftest);

                let ps: Vec<_> = (1..=3).collect();
                let me = ps[i];
                let mut next = ps[(i + 1) % 3];
                let mut last = ps[(i + 2) % 3];
                if me == 2 {
                    std::mem::swap(&mut next, &mut last);
                }

                cote.init(sid, next, me < next).await?;
                cote.init(sid, last, me < last).await?;

                let mut res = Vec::new();
                for n in [next, last] {
                    if me < n {
                        let alphas = vec![FF2_128::one(); 2];
                        let ta = cote.send(sid, n, alphas).await?;
                        res.push(ta);
                    } else {
                        let selections = vec![true, false];
                        let tb = cote.recv(sid, n, selections).await?;
                        res.push(tb);
                    }
                }

                Ok((me, res))
            });
        }

        let mut res: HashMap<usize, _> = HashMap::with_capacity(3);

        while let Some(r) = js.join_next().await {
            let (id, rs) = r.unwrap().unwrap();
            res.insert(id.into(), rs);
        }

        let z = FF2_128::zero();
        let o = FF2_128::one();

        let rs_1 = &res[&1];
        let rs_2 = &res[&2];
        let rs_3 = &res[&3];

        assert!((rs_1[0][0] + rs_2[0][0]) == o);
        assert!((rs_1[0][1] + rs_2[0][1]) == z);

        assert!((rs_1[1][0] + rs_3[0][0]) == o);
        assert!((rs_1[1][1] + rs_3[0][1]) == z);

        assert!((rs_2[1][0] + rs_3[1][0]) == o);
        assert!((rs_2[1][1] + rs_3[1][1]) == z);
    }
}
