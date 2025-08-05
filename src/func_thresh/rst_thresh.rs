use crate::{
    auth_bits::ThreshAbits,
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    circuits::{elements::out_mask, CircuitElement, TCircuit},
    field::{Field, RandElement},
    func_mpc::AsyncMpc,
    func_thresh::{AsyncThresh, InputId},
    func_thresh_abit::AsyncTabit,
    party::PartyId,
};

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::Context;
use log::trace;

use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct RstThreshPlayer<T: Field, FM, FT> {
    party_id: PartyId,
    //n: u16,
    //t: u16,
    parties: Vec<PartyId>,
    party_points: Vec<T>,
    mpc: Arc<FM>,
    tabit: Arc<FT>,
    cur_mpc_sid: AtomicU64,
    delta: RefCell<Option<T>>,
    tabits: RefCell<Vec<ThreshAbits<T>>>,
    input_tabit_idx: RefCell<HashMap<InputId, usize>>,
}

#[derive(Serialize, Deserialize)]
struct State<T> {
    delta: T,
    tabits: Vec<ThreshAbits<T>>,
    tabit_idx: HashMap<InputId, usize>,
}

impl<T: Field, FM, FT> BaseFunc for RstThreshPlayer<T, FM, FT> {
    const FUNC_ID: FuncId = FuncId::Fthresh;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fmpc, FuncId::Ftabit];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

impl<T, FM: AsyncMpc<T>, FT: AsyncTabit<T>> AsyncThresh for RstThreshPlayer<T, FM, FT>
where
    T: Field + CircuitElement + RandElement + Copy + Serialize,
    for<'d> T: Deserialize<'d>,
{
    /// Write out the state of any authenticated shares
    fn write_state_to_file(&self, file_name: &str) -> Result<(), UnexpectedError> {
        let state = {
            let delta = self.delta.borrow().unwrap().clone();
            let tabits = self.tabits.borrow().clone();
            let tabit_idx = self.input_tabit_idx.borrow().clone();

            State {
                delta,
                tabits,
                tabit_idx,
            }
        };

        let sid = SessionId::new(FuncId::Fthresh);

        let f = File::create(file_name)
            .with_context(|| self.err(sid, "Failed to create state file"))?;

        serde_json::to_writer(f, &state)
            .with_context(|| self.err(sid, "Failed to write state file"))?;

        Ok(())
    }

    /// Restore state from the file, reading the MAC key and any authenticated shares
    async fn resume_from_state_file(
        &mut self,
        file_name: &str,
        run_init: bool,
    ) -> Result<(), UnexpectedError> {
        let sid = SessionId::new(FuncId::Fthresh);

        let f =
            File::open(file_name).with_context(|| self.err(sid, "Failed to open state file"))?;

        let state: State<T> = serde_json::from_reader(f)
            .with_context(|| self.err(sid, "Failed to deserialize state file"))?;

        {
            let mut d = self.delta.borrow_mut();
            *d = Some(state.delta.clone());
        }

        {
            let mut t = self.tabits.borrow_mut();
            *t = state.tabits;
        }

        {
            let mut i = self.input_tabit_idx.borrow_mut();
            *i = state.tabit_idx;
        }

        if run_init {
            let sid = SessionId::new(FuncId::Fthresh);
            self.tabit
                .init(sid, state.delta)
                .await
                .with_context(|| self.err(sid, "Failed to initialize Ftabit"))?;
        }

        Ok(())
    }

    async fn init(&mut self) -> Result<(), UnexpectedError> {
        // Sample the correlation used between the thresh abit / mpc protocols
        let delta = {
            let mut rng = rand::thread_rng();
            T::rand(&mut rng)
        };

        {
            let mut d = self.delta.borrow_mut();
            *d = Some(delta);
        }

        let sid = SessionId::new(FuncId::Fthresh);
        trace!("{}: init ({sid})", self.party_id);
        // run the tabit setup, mostly the OT extension
        // OT extension could hypothetically be shared with
        // the MPC protocol, but is not done for now
        self.tabit
            .init(sid, delta)
            .await
            .with_context(|| self.err(sid, "Failed to initialize Ftabit"))?;

        Ok(())
    }

    async fn setup<I, O: CircuitElement>(
        &mut self,
        input: &[bool],
        out_ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<(), UnexpectedError> {
        let delta = { self.delta.borrow().clone().unwrap() };

        let sid = SessionId::new(FuncId::Fthresh);
        let mpc_sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let mpc_sid = SessionId {
            parent: FuncId::Fthresh,
            id: mpc_sid,
        };
        self.mpc
            .init(mpc_sid, Some(delta))
            .await
            .with_context(|| self.err(sid, format!("Failed to initialize Fmpc {mpc_sid}")))?;

        assert!(out_ids.len() == circuit.outputs.len());
        let mut tabits = self
            .tabit
            .sample(sid, out_ids.len())
            .await
            .with_context(|| {
                self.err(
                    sid,
                    format!("Failed to sample new thresh abits with ids {out_ids:?}"),
                )
            })?;

        let my_point = T::from(self.party_id.into());
        let idx: Vec<usize> = (0..out_ids.len()).collect();
        let abits = tabits.convert(&my_point, &self.party_points, &self.party_points, &idx);

        self.mpc
            .input_abit(mpc_sid, abits)
            .await
            .with_context(|| self.err(sid, format!("Failed to provide abits to Fmpc {mpc_sid}")))?;

        let size = input.len();
        // Simplify for now, assume all parties have the  same number of inputs
        assert!(circuit.inputs.len() == size * self.parties.len());
        for &p in self.parties.iter() {
            let input = if p == self.party_id {
                Some(input.to_vec())
            } else {
                None
            };
            self.mpc
                .input_multi(mpc_sid, p, size, input)
                .await
                .with_context(|| {
                    self.err(
                        sid,
                        format!("Failed to provide input from {p} to Fmpc {mpc_sid}"),
                    )
                })?;
        }

        let mask_circuit = out_mask(circuit);

        // mpc outputs the masked output bits (s + r)
        let masked_out = self
            .mpc
            .eval_pub(mpc_sid, &self.parties, &mask_circuit)
            .await
            .with_context(|| {
                self.err(
                    sid,
                    format!("Failed to evaluate setup circuit in Fmpc {mpc_sid}"),
                )
            })?;

        let mut bits = vec![false; O::BIT_SIZE];
        masked_out.to_bits(&mut bits[..]);
        // current tabits are <<r>> calculate <<s>> = (s+r) + <<r>>
        tabits.add_consts(&bits, delta);

        {
            let mut prev = self.tabits.borrow_mut();
            prev.push(tabits);
        }

        {
            let mut map = self.input_tabit_idx.borrow_mut();
            let next_idx = map.len();
            map.extend(out_ids.iter().cloned().zip(next_idx..));
        }

        Ok(())
    }

    async fn sample(&mut self, ids: &[InputId]) -> Result<(), UnexpectedError> {
        let sid = SessionId::new(FuncId::Fthresh);
        let tabits = self.tabit.sample(sid, ids.len()).await.with_context(|| {
            self.err(
                sid,
                format!("Failed to sample new thresh abits with ids {ids:?}"),
            )
        })?;

        {
            let mut prev = self.tabits.borrow_mut();
            prev.push(tabits);
        }

        {
            let mut map = self.input_tabit_idx.borrow_mut();
            let next_idx = map.len();
            map.extend(ids.iter().cloned().zip(next_idx..));
        }

        Ok(())
    }

    async fn eval<I, O: CircuitElement>(
        &mut self,
        parties: &[PartyId],
        ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<O, UnexpectedError> {
        assert!(ids.len() == circuit.inputs.len());

        let delta = { self.delta.borrow().clone().unwrap() };

        let sid = SessionId::new(FuncId::Fthresh);
        let mpc_sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let mpc_sid = SessionId {
            parent: FuncId::Fthresh,
            id: mpc_sid,
        };
        self.mpc
            .init(mpc_sid, Some(delta))
            .await
            .with_context(|| self.err(sid, format!("Failed to initialize Fmpc {mpc_sid}")))?;

        let all_ids: Vec<_> = {
            let map = self.input_tabit_idx.borrow();
            ids.iter().map(|&i| map[&i].clone()).collect()
        };

        // Convert the saved tabits into t-party abits
        let abits = {
            let all_tabits = self.tabits.borrow();

            let my_point = T::from(self.party_id.into());
            let all_points = &self.party_points;
            let sub_points: Vec<_> = parties.iter().map(|p| T::from(*p as u64)).collect();

            let mut start = 0;
            // TODO: this doesn't work well if the ids splice between multiple sets of abits
            all_tabits
                .iter()
                .map(|ts| {
                    let end = start + ts.nbits;
                    let idx: Vec<_> = all_ids
                        .iter()
                        .filter(|&&i| i >= start && i < end)
                        .map(|&i| i - start)
                        .collect();
                    start = end;
                    ts.convert(&my_point, all_points, &sub_points, &idx)
                })
                .reduce(|mut a, b| {
                    a.append(b);
                    a
                })
                .unwrap()
        };

        self.mpc
            .input_abit(mpc_sid, abits)
            .await
            .with_context(|| self.err(sid, format!("Failed to provide abits to Fmpc {mpc_sid}")))?;

        let out = self
            .mpc
            .eval_pub(mpc_sid, &parties, circuit)
            .await
            .with_context(|| {
                self.err(
                    sid,
                    format!("Failed to evaluate the circuit in Fmpc {mpc_sid}"),
                )
            })?;

        Ok(out)
    }
}

impl<T: Field, FM: AsyncMpc<T>, FT: AsyncTabit<T>> RstThreshPlayer<T, FM, FT> {
    pub fn new(
        party_id: PartyId,
        n: PartyId,
        _t: u16,
        mpc: Arc<FM>,
        tabit: Arc<FT>,
    ) -> Result<Self, ()> {
        let parties: Vec<PartyId> = (1..n + 1).collect();
        let party_points: Vec<T> = (1..n + 1).map(|x| T::from(x.into())).collect();

        Ok(RstThreshPlayer {
            party_id: party_id,
            //n: n,
            //t: t,
            tabits: RefCell::new(Vec::new()),
            input_tabit_idx: RefCell::new(HashMap::new()),
            parties,
            party_points,
            mpc: mpc,
            tabit: tabit,
            cur_mpc_sid: AtomicU64::new(0),
            delta: RefCell::new(None),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        circuits::{arith::sum_circuit, Gate, TCircuit},
        ff2_128::FF2_128,
        func_abit::tests::build_test_abits,
        func_com::tests::build_test_coms,
        func_cote::kos_cote::tests::build_test_cotes,
        func_mpc::tests::build_test_mpcs,
        func_mult::tests::build_test_mults,
        func_net::tests::{build_test_nets, get_test_party_infos},
        func_rand::tests::build_test_rands,
        func_thresh_abit::tests::build_test_tabits,
    };
    use tokio::io;
    use tokio::task::JoinSet;

    pub fn build_test_threshs<FM: AsyncMpc<FF2_128>, FT: AsyncTabit<FF2_128>>(
        mpcs: &[Arc<FM>],
        tabits: &[Arc<FT>],
    ) -> Vec<RstThreshPlayer<FF2_128, FM, FT>> {
        let num = mpcs.len();
        (1..=num)
            .map(|i| {
                RstThreshPlayer::new(
                    i as PartyId,
                    num as PartyId,
                    (num - 1) as PartyId,
                    mpcs[i - 1].clone(),
                    tabits[i - 1].clone(),
                )
                .unwrap()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_full_execution() -> io::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let party_info = get_test_party_infos(3);
        let nets = build_test_nets(
            &party_info,
            vec![FuncId::Fcom, FuncId::Fcote, FuncId::Fmult, FuncId::Ftabit],
        )
        .await;
        let mpcs = build_test_mpcs(&party_info);
        let coms = build_test_coms(&nets);
        let rands = build_test_rands(&coms);
        let abits = build_test_abits(&party_info);
        let cotes = build_test_cotes(&nets, &party_info);
        let mults = build_test_mults(&nets, &cotes);
        let tabits = build_test_tabits(&abits, &rands, &mults, &coms, &nets);
        let threshs = build_test_threshs(&mpcs, &tabits);

        let mut js = JoinSet::new();

        for (i, t) in threshs.into_iter().enumerate() {
            js.spawn(async move {
                // can't have a normal reference and await as that will trigger Send errors
                // but also if I have the mutable reference I don't need refcell at all
                let mut t = t;
                t.init().await?;

                let mut input = vec![false; 128];
                input[i] = true;

                let out_ids: Vec<_> = (0..128).collect();

                let sum_c = sum_circuit::<FF2_128>(3);

                t.setup(&input, &out_ids, &sum_c).await?;

                let c: TCircuit<FF2_128, FF2_128> = {
                    let inputs = (0..128).collect();
                    let gates = (0..128)
                        .map(|i| Gate::Xor(i, 127 - i, 128 + i))
                        .chain((0..128).map(|i| Gate::And(128 + i, 128 + i, 256 + i)))
                        .collect();
                    let outputs = (256..384).collect();

                    TCircuit::from_parts(inputs, gates, outputs)
                };

                let parties: Vec<_> = (1..=3).collect();

                let out = t.eval(&parties, &out_ids, &c).await;

                out
            });
        }

        while let Some(r) = js.join_next().await {
            let s = r.unwrap().unwrap();
            assert_eq!(s, FF2_128::new(7 << 61, 7));
        }

        Ok(())
    }
}
