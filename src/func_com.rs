use crate::{
    base_func::{BaseFunc, CheatOrUnexpectedError, CheatDetectedError, FuncId, SessionId, UnexpectedError},
    func_net::AsyncNet,
    party::PartyId,
};

use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use anyhow::Context;

#[derive(Debug)]
pub struct FolkloreComPlayer<FN> {
    party_id: PartyId,
    //n: u16,
    net: Arc<FN>,
    commit_seeds: Mutex<HashMap<(PartyId, SessionId), [u8; 32]>>,
    commitments: Mutex<HashMap<(PartyId, SessionId), [u8; 32]>>,
}

impl<FN> BaseFunc for FolkloreComPlayer<FN> {
    const FUNC_ID: FuncId = FuncId::Fcom;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fnet];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DecomError {
    #[error("Decommitment invalid: {0}")]
    CheatDetected(#[from] CheatDetectedError),
    #[error(transparent)]
    Unexpected(#[from] UnexpectedError),
}

impl From<anyhow::Error> for DecomError {
    fn from(e: anyhow::Error) -> Self {
        DecomError::Unexpected(e.into())
    }
}

/// A trait to represent the F_Com functionality
pub trait AsyncCom {
    /// Commit a value `data` to `party`.
    fn commit_to<D: AsRef<[u8]>>(
        &self,
        sid: SessionId,
        party: PartyId,
        data: D,
    ) -> impl Future<Output = Result<(), UnexpectedError>>;

    /// Receive the commitment from `party`
    fn expect_from(
        &self,
        sid: SessionId,
        party: PartyId,
    ) -> impl Future<Output = Result<(), UnexpectedError>>;

    /// Decommit the previously-committed value to `party`
    fn decommit_to<D: AsRef<[u8]>>(
        &self,
        sid: SessionId,
        party: PartyId,
        data: D,
    ) -> impl Future<Output = Result<(), UnexpectedError>>;

    /// Receive the decommitment from `party` and verify its correctness
    fn value_from(
        &self,
        sid: SessionId,
        party: PartyId,
        recv_sz: usize,
    ) -> impl Future<Output = Result<Vec<u8>, CheatOrUnexpectedError>>;
}

impl<N: AsyncNet + Sync + 'static> AsyncCom for FolkloreComPlayer<N> {
    async fn commit_to<D: AsRef<[u8]>>(
        &self,
        sid: SessionId,
        party: PartyId,
        data: D,
    ) -> Result<(), UnexpectedError> {
        // Take in the input data, sample some randomness, and send a commitment string to `party'
        let seed: [u8; 32] = rand::random::<[u8; 32]>();

        {
            let mut guard_seeds = self.commit_seeds.lock().unwrap();
            (*guard_seeds).insert((party, sid), seed);
        }

        let commitment = h(&[&seed, data.as_ref()]);

        self.net
            .clone()
            .send_to_local(party, FuncId::Fcom, sid, commitment)
            .await
            .with_context(|| self.err(sid, "Failed to send commitment to {party}"))?;

        Ok(())
    }

    async fn expect_from(
        &self,
        sid: SessionId,
        party: PartyId,
    ) -> Result<(), UnexpectedError> {
        // receive the commitment string from `party'
        let com = Arc::from([0; 32].as_slice());
        let (com, read) = self
            .net
            .clone()
            .recv_from(party, FuncId::Fcom, sid, com)
            .await
            .with_context(|| self.err(sid, "Failed to receive commitment from {party}"))?;

        if read != 32 {
            return Err(self.unexpected(
                sid,
                format!("Commitment from {party} too small {read} < 32"),
            ));
        }

        {
            let com: [u8; 32] = com.as_ref().try_into().unwrap();
            let mut guard_comms = self.commitments.lock().unwrap();
            (*guard_comms).insert((party, sid), com);
        }

        Ok(())
    }

    async fn decommit_to<D: AsRef<[u8]>>(
        &self,
        sid: SessionId,
        party: PartyId,
        data: D,
    ) -> Result<(), UnexpectedError> {
        // Take in the data and id, look up the randomness used to commit originally, and send both to `party'
        let seed = {
            let guard_seeds = self.commit_seeds.lock().unwrap();
            let seed_opt = (*guard_seeds).get(&(party, sid));
            match seed_opt {
                Some(x) => Arc::from(&x[..]),
                None => {
                    return Err(self.unexpected(sid, format!("no commitment found for {party}")));
                }
            }
        };

        // TODO: this should be one send instead of 2
        self.net
            .clone()
            .send_to(party, FuncId::Fcom, sid, seed)
            .await
            .with_context(|| self.err(sid, format!("Failed to send seed to {party}")))?;
        self.net
            .clone()
            .send_to_local(party, FuncId::Fcom, sid, data)
            .await
            .with_context(|| self.err(sid, format!("Failed to send data to {party}")))?;

        Ok(())
    }

    async fn value_from(
        &self,
        sid: SessionId,
        party: PartyId,
        recv_sz: usize,
    ) -> Result<Vec<u8>, CheatOrUnexpectedError> {
        // Receive the randomness and data and verify that it matches with the commitment received before
        let comm = {
            let guard_comms = self.commitments.lock().unwrap();
            let comm_opt = (*guard_comms).get(&(party, sid));
            match comm_opt {
                Some(x) => *x,
                None => {
                    return Err(self
                        .unexpected(sid, format!("no commitment found for {party}"))
                        .into())
                }
            }
        };

        // Receive the randomness
        let seed = Arc::from([0; 32].as_slice());
        let (seed, _) = self
            .net
            .clone()
            .recv_from(party, FuncId::Fcom, sid, seed)
            .await
            .with_context(|| self.err(sid, format!("Failed to receive decom seed from {party}")))?;

        // Receive the data
        let data = unsafe { Arc::new_zeroed_slice(recv_sz).assume_init() };
        let (data, _) = self
            .net
            .clone()
            .recv_from(party, FuncId::Fcom, sid, data)
            .await
            .with_context(|| self.err(sid, format!("Failed to receive decom data from {party}")))?;

        let comm_check = h(&[&seed, &data]);
        if comm == comm_check {
            Ok(data.as_ref().to_vec())
        } else {
            Err(self
                .cheat(sid, Some(party), "Decommitment invalid".into())
                .into())
        }
    }
}

impl<N: AsyncNet> FolkloreComPlayer<N> {
    pub fn new(party_id: PartyId, _n: PartyId, net: Arc<N>) -> Result<Self, ()> {
        Ok(FolkloreComPlayer {
            party_id: party_id,
            //n: n,
            net: net,
            commit_seeds: Mutex::new(HashMap::new()),
            commitments: Mutex::new(HashMap::new()),
        })
    }
}

fn h(inputs: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    let mut buff = [0u8; 32];

    for input in inputs {
        hasher.update(input);
    }

    let hash = hasher.finalize();
    buff.copy_from_slice(&hash);

    return buff;
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::func_net::tests::{build_test_nets, get_test_party_infos};
    use tokio::io;

    pub fn build_test_coms<N: AsyncNet>(nets: &[Arc<N>]) -> Vec<Arc<FolkloreComPlayer<N>>> {
        let num = nets.len();
        (1..=num)
            .map(|i| {
                Arc::new(
                    FolkloreComPlayer::new(i as PartyId, num as PartyId, nets[i - 1].clone())
                        .unwrap(),
                )
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_com() -> io::Result<()> {
        let party_info = get_test_party_infos(2);
        let nets = build_test_nets(&party_info, vec![FuncId::Fcom]).await;
        let coms = build_test_coms(&nets);

        let com1 = coms[0].clone();
        let com2 = coms[1].clone();

        let sid = SessionId {
            parent: FuncId::Ftest,
            id: 1,
        };

        let h1 = tokio::spawn(async move {
            let bytes: Arc<[u8]> = Arc::from([1, 2, 3, 4]);
            let r1 = com1.clone().commit_to(sid, 2, bytes.clone()).await;
            assert!(r1.is_ok());
            let r2 = com1.decommit_to(sid, 2, bytes).await;
            assert!(r2.is_ok());
        });

        let h2 = tokio::spawn(async move {
            let r1 = com2.clone().expect_from(sid, 1).await;
            assert!(r1.is_ok());
            let r2 = com2.value_from(sid, 1, 4).await;
            assert!(r2.is_ok());
            assert!(&r2.unwrap()[..4] == &[1, 2, 3, 4]);
        });

        h1.await?;
        h2.await?;

        Ok(())
    }
}
