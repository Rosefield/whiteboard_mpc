use crate::{
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    common_protos::broadcast_commit_open,
    func_com::AsyncCom,
    party::PartyId,
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Context;

use rand::Rng;
use rand_chacha;
use rand_core::SeedableRng;

#[derive(Debug)]
pub struct FolkloreRandPlayer<FC> {
    party_id: PartyId,
    n: u16,
    com: Arc<FC>,
    rngs: Mutex<HashMap<SessionId, rand_chacha::ChaCha20Rng>>,
}

impl<FC> BaseFunc for FolkloreRandPlayer<FC> {
    const FUNC_ID: FuncId = FuncId::Frand;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fcom];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// Trait to represent coin tossing, where a group of parties
/// will agree upon a shared random value.
pub trait AsyncRand {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId) -> Result<(), UnexpectedError>;

    /// Sample the next set of bytes
    async fn rand(&self, sid: SessionId, num_bytes: usize) -> Result<Vec<u8>, UnexpectedError>;
}

impl<FC: AsyncCom> AsyncRand for FolkloreRandPlayer<FC> {
    async fn init(&self, sid: SessionId) -> Result<(), UnexpectedError> {
        let mut rand_val = [0; 32];
        {
            let mut rng = rand::thread_rng();
            rng.fill(&mut rand_val);
        }

        let ssid = sid.derive_ssid(FuncId::Frand);

        let parties: Vec<_> = (1..=self.n).collect();
        let other_seeds =
            broadcast_commit_open(ssid, &rand_val, self.party_id, &parties, self.com.clone())
                .await
                .with_context(|| self.err(sid, "failed to initialize seeds"))?;

        let mut joint = rand_val;
        other_seeds.into_iter().for_each(|other| {
            joint
                .iter_mut()
                .zip(other.into_iter())
                .for_each(|(x, y)| *x ^= y);
        });

        {
            let mut guard_rngs = self.rngs.lock().unwrap();
            (*guard_rngs).insert(sid, rand_chacha::ChaCha20Rng::from_seed(joint));
        }

        Ok(())
    }

    /// Authenticate shares from each party
    async fn rand(&self, sid: SessionId, num_bytes: usize) -> Result<Vec<u8>, UnexpectedError> {
        let mut guard_rngs = self.rngs.lock().unwrap();
        let rng = (*guard_rngs).get_mut(&sid).unwrap();

        let mut rand_nums: Vec<u8> = vec![0; num_bytes];
        rng.fill(&mut rand_nums[..]);

        Ok(rand_nums)
    }
}

impl<FC> FolkloreRandPlayer<FC> {
    pub fn new(party_id: PartyId, n: PartyId, com: Arc<FC>) -> Result<Self, ()> {
        Ok(FolkloreRandPlayer {
            party_id: party_id,
            n: n,
            com: com,
            rngs: Mutex::new(HashMap::new()),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    //use crate::{func_net::tests::get_test_party_infos, ff2_128::FF2_128};
    use crate::func_com::tests::build_test_coms;
    use crate::func_net::tests::{build_test_nets, get_test_party_infos};
    use tokio::io;

    pub fn build_test_rands<FC: AsyncCom>(coms: &[Arc<FC>]) -> Vec<Arc<FolkloreRandPlayer<FC>>> {
        let num = coms.len();
        (1..=num)
            .map(|i| {
                Arc::new(
                    FolkloreRandPlayer::new(i as PartyId, num as PartyId, coms[i - 1].clone())
                        .unwrap(),
                )
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_init_rand() -> io::Result<()> {
        let party_info = get_test_party_infos(3);
        let nets = build_test_nets(&party_info, vec![FuncId::Fcom]).await;
        let coms = build_test_coms(&nets);
        let rands = build_test_rands(&coms);

        let rands1 = rands[0].clone();
        let rands2 = rands[1].clone();
        let rands3 = rands[2].clone();

        let h1 = tokio::spawn(async move {
            let sid = SessionId::new(FuncId::Ftest);
            let r1 = rands1.init(sid).await;
            assert!(r1.is_ok());

            let r1b = rands1.rand(sid, 3).await;
            assert!(r1b.is_ok());

            let r1c = rands1.rand(sid, 3).await;
            assert!(r1c.is_ok());

            return (r1b.unwrap(), r1c.unwrap());
        });

        let h2 = tokio::spawn(async move {
            let sid = SessionId::new(FuncId::Ftest);
            let r2 = rands2.init(sid).await;
            assert!(r2.is_ok());

            let r2b = rands2.rand(sid, 3).await;
            assert!(r2b.is_ok());

            return r2b.unwrap();
        });

        let h3 = tokio::spawn(async move {
            let sid = SessionId::new(FuncId::Ftest);
            let r3 = rands3.init(sid).await;
            assert!(r3.is_ok());

            let r3b = rands3.rand(sid, 3).await;
            assert!(r3b.is_ok());

            return r3b.unwrap();
        });

        let v1 = h1.await?;
        let v2 = h2.await?;
        let v3 = h3.await?;

        assert_eq!(v1.0, v2);
        assert_eq!(v2, v3);
        assert_eq!(v1.0.len(), 3);

        assert_ne!(v1.0, v1.1);

        Ok(())
    }
}
