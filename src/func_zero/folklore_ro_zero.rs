use crate::{
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError, CheatOrUnexpectedError},
    party::PartyId,
    field::{Group, RandElement, FWrap},
    func_com::AsyncCom,
    func_zero::AsyncZeroShare,
    ro::RO,
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use futures::stream::{StreamExt, FuturesUnordered};
use rand::Rng;

struct SessionState {
    idx: usize,
    ros: HashMap<PartyId, RO>
}

pub struct FolkloreRoZeroPlayer<FC> {
    party_id: PartyId,
    com: Arc<FC>,
    state: Mutex<HashMap<SessionId, SessionState>>
}

impl<FC> BaseFunc for FolkloreRoZeroPlayer<FC> {
    const FUNC_ID: FuncId = FuncId::Fzero;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fcom];

    fn party(&self) -> PartyId {
        self.party_id
    }
}


impl<FC: AsyncCom> AsyncZeroShare for FolkloreRoZeroPlayer<FC> {
    async fn init(&self, sid: SessionId, parties: &[PartyId]) -> Result<(), CheatOrUnexpectedError> {
        assert!(parties.contains(&self.party_id));

        let comsid = sid.derive_ssid(FuncId::Fzero);
        let comsid2 = comsid.next();

        let mut set = FuturesUnordered::new();

        for p in parties.iter().filter(|p| **p != self.party_id) {
            let mut rng = rand::thread_rng();
            let mut seed = [0u8; 16];
            rng.fill(&mut seed[..]);

            let [comsid, comsid2] = if self.party_id < *p {
                [comsid, comsid2]
            } else {
                [comsid2, comsid]
            };

            set.push({
                let p = *p;
                async move {
                    let comf = self.com.commit_to(comsid, p, &seed);
                    let expf = self.com.expect_from(comsid2, p);

                    let (r1, r2) = tokio::join!(comf, expf);
                    r1?;
                    r2?;

                    let decf = self.com.decommit_to(comsid, p, &seed);
                    let valf = async {
                        self.com.value_from(comsid2, p, 16).await.map(|bs| bs.try_into().unwrap())
                    };

                    let (r3, r4) = tokio::join!(decf, valf);

                    r3?;
                    let seed2 = r4?;

                    let seed = FWrap(seed) ^ FWrap(seed2);

                    let ro = RO::new()
                                    .add_context(FuncId::Fzero.as_bytes())
                                    .add_context(sid.as_bytes())
                                    .add_context(seed.0);

                    Result::<_, CheatOrUnexpectedError>::Ok((p, ro))
                }
            });
        }


        let mut state = SessionState {
            idx: 0,
            ros: HashMap::new()
        };

        while let Some(res) = set.next().await {
            let (party, ro) = res?;

            state.ros.insert(party, ro);
        }

        self.state.lock().unwrap()
            .insert(sid, state);


        Ok(())
    }

    fn generate_noninteractive<G: Group + RandElement>(&self, sid: SessionId) -> Result<G, UnexpectedError> {

        let mut state = self.state.lock().unwrap();

        let Some(s) = state.get_mut(&sid) else {
            return Err(self.unexpected(sid, format!("Init has not been run for {sid}")));
        };

        let idx = s.idx;

        let val = s.ros.iter().map(|(k,v)| {

            let mut val: G = v.generate(idx.to_le_bytes());

            // want contributions to cancel out, so one of the parties negates their share
            if self.party_id > *k {
                val = -val;
            }

            val
        }).sum();

        s.idx += 1;

        Ok(val)
    }
}

impl<FC> FolkloreRoZeroPlayer<FC> {
    pub fn new(party_id: PartyId, com: Arc<FC>) -> Self {
        Self {
            party_id,
            com,
            state: Mutex::new(HashMap::new())
        }
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        func_net::tests::{build_test_nets, get_test_party_infos},
        func_com::tests::build_test_coms,
        ff2_128::FF2_128,
        field::ConstInt,
    };

    use std::sync::Arc;

    pub fn build_test_zeros<FC>(
        coms: &[Arc<FC>],
    ) -> Vec<Arc<FolkloreRoZeroPlayer<FC>>> {
        let num = coms.len() as PartyId;
        (1..=num)
            .map(|i| {
                let idx = (i-1) as usize;
                Arc::new(FolkloreRoZeroPlayer::new(i,  coms[idx].clone()))
            })
            .collect()
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_zero_share() -> Result<(), CheatOrUnexpectedError>{
        let parties = get_test_party_infos(2);
        let funcs = vec![FuncId::Fcom];
        let nets = build_test_nets(&parties, funcs).await;
        let coms = build_test_coms(&nets);
        let zeros = build_test_zeros(&coms);

        let sid = SessionId::new(FuncId::Ftest);

        let h1 = tokio::spawn({
            let zero = zeros[0].clone();
            async move {
                zero.init(sid, &[1,2]).await
                    .map(|_| zero.generate_noninteractive::<FF2_128>(sid).unwrap())
            }
        });

        let h2 = tokio::spawn({
            let zero = zeros[1].clone();
            async move {
                zero.init(sid, &[1,2]).await
                    .map(|_| zero.generate_noninteractive::<FF2_128>(sid).unwrap())
            }
        });

        let v1 = h1.await.unwrap()?;
        let v2 = h2.await.unwrap()?;

        assert_eq!(v1+v2, FF2_128::zero());

        Ok(())
    }

}
