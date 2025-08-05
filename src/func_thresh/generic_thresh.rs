use crate::{
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    circuits::{
        generic_thresh::{add_validation_project, init_circuit, sample_circuit, setup_circuit},
        CircuitCollection, CircuitElement, CircuitRing, TCircuit,
    },
    field::{Field, RandElement},
    func_mpc::AsyncMpc,
    func_thresh::{AsyncThresh, InputId},
    party::PartyId,
    polynomial::lagrange_poly,
};

use std::{
    collections::HashMap,
    fs::File,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use rand::Rng;

use anyhow::{anyhow, Context};

use serde::{Deserialize, Serialize};
use serde_json;

use log::trace;

#[derive(Debug)]
pub struct GenericThreshPlayer<T, FM> {
    party_id: PartyId,
    n: u16,
    t: u16,
    state: Option<State<T>>,
    parties: Vec<PartyId>,
    party_points: Vec<T>,
    lp_coeff: T,
    mpc: Arc<FM>,
    cur_mpc_sid: AtomicU64,
}

#[derive(Serialize, Deserialize, Debug)]
struct State<T> {
    // Our share of the global mac key
    share_alpha: T,
    // our authenticated shares of (x, ax)
    share_inputs: Vec<(T, T)>,
    // maps an input id to the (idx, bit) of the shares
    // so (2, 100) would be the 101st bit of share_inputs[2]
    input_share_idx: HashMap<InputId, (usize, usize)>,
}

impl<T, FM> BaseFunc for GenericThreshPlayer<T, FM> {
    const FUNC_ID: FuncId = FuncId::Fthresh;
    const REQUIRED_FUNCS: &'static [FuncId] = &[FuncId::Fmpc];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/*
pub trait Foo: CircuitElement {
    fn x() -> TCircuit<(Self, Self), Self>;
    fn y() -> TCircuit<(Self, Self), Self>;
}
*/

// Compiler sometimes crashes if this is T: CircuitRing for some reason, but works with T: Foo which is identical to CircuitRing :shrug:
impl<T, FM: AsyncMpc<T>> AsyncThresh for GenericThreshPlayer<T, FM>
where
    T: Field + RandElement + CircuitRing + Copy + Serialize,
    for<'d> T: Deserialize<'d>,
{
    /// Write out the state of any authenticated shares
    fn write_state_to_file(&self, state_file: &str) -> Result<(), UnexpectedError> {
        if self.state.is_none() {
            return Err(anyhow!("Init/resume has not been run, no state to write").into());
        }

        let state = self.state.as_ref().unwrap();

        let sid = SessionId::new(FuncId::Fthresh);

        let f = File::create(state_file)
            .with_context(|| self.err(sid, "Failed to create state file"))?;

        serde_json::to_writer(f, state)
            .with_context(|| self.err(sid, "Failed to write state file"))?;

        Ok(())
    }

    /// Restore state from the file, reading the MAC key and any authenticated shares
    async fn resume_from_state_file(
        &mut self,
        state_file: &str,
        _run_init: bool,
    ) -> Result<(), UnexpectedError> {
        let sid = SessionId::new(FuncId::Fthresh);

        let f =
            File::open(state_file).with_context(|| self.err(sid, "Failed to open state file"))?;

        let state: State<T> = serde_json::from_reader(f)
            .with_context(|| self.err(sid, "Failed to deserialize state file"))?;

        self.state = Some(state);

        Ok(())
    }

    async fn init(&mut self) -> Result<(), UnexpectedError> {
        if self.state.is_some() {
            return Err(anyhow!("Init run after already initialized").into());
        }

        let sid = SessionId::new(FuncId::Fthresh);
        let mpc_sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let mpc_sid = SessionId {
            parent: FuncId::Fthresh,
            id: mpc_sid,
        };

        let _ = self.mpc.init(mpc_sid, None).await?;

        for &i in self.parties.iter() {
            if i == self.party_id {
                let shares_alpha_i: Vec<_> = {
                    let mut rng = rand::thread_rng();
                    let shares_alpha_i = (0..self.t).map(|_| T::rand(&mut rng)).collect();
                    shares_alpha_i
                };
                let _ = self
                    .mpc
                    .input_multi(mpc_sid, i, self.t.into(), Some(shares_alpha_i))
                    .await;
            } else {
                let _ = self
                    .mpc
                    .input_multi::<Vec<T>>(mpc_sid, i, self.t.into(), None)
                    .await;
            }
        }

        let circuit: TCircuit<Vec<T>, Vec<T>> =
            init_circuit(self.n.into(), self.t.into(), &self.party_points);

        trace!("running mpc eval to generate shares of alpha");

        let share_alpha = self
            .mpc
            .eval_priv(mpc_sid, &self.parties, &circuit)
            .await
            .with_context(|| self.err(sid, "Failed to generate alpha shares"))?;

        let state = State {
            share_alpha,
            share_inputs: Vec::new(),
            input_share_idx: HashMap::new(),
        };

        self.state = Some(state);

        Ok(())
    }

    async fn setup<I, O: CircuitElement>(
        &mut self,
        input: &[bool],
        out_ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<(), UnexpectedError> {
        if self.state.is_none() {
            return Err(anyhow!("Init/resume not run").into());
        }
        let in_size = input.len();
        // Simplify for now, assume all parties have the  same number of inputs
        assert!(circuit.inputs.len() == in_size * self.parties.len());
        assert!(circuit.outputs.len() == out_ids.len());
        let alpha = self.state.as_ref().unwrap().share_alpha.clone();

        let sid = SessionId::new(FuncId::Fthresh);
        let mpc_sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let mpc_sid = SessionId {
            parent: FuncId::Fthresh,
            id: mpc_sid,
        };
        self.mpc
            .init(mpc_sid, None)
            .await
            .with_context(|| self.err(sid, format!("Failed to initialize Fmpc {mpc_sid}")))?;

        for &p in self.parties.iter() {
            let input = if p == self.party_id {
                Some(input.to_vec())
            } else {
                None
            };
            self.mpc
                .input_multi(mpc_sid, p, in_size, input)
                .await
                .with_context(|| {
                    self.err(
                        sid,
                        format!("Failed to provide input from {p} to Fmpc {mpc_sid}"),
                    )
                })?;
        }

        let out_size = circuit.outputs.len();
        let num_shares = (out_size + T::BIT_SIZE - 1) / T::BIT_SIZE;
        let fs_size: usize = 2 * num_shares * (self.t as usize - 1);

        for &i in self.parties.iter() {
            if i == self.party_id {
                // make 2*l random degree t-1 polynomials to use as zero shares
                let all_fs: Vec<T> = {
                    let mut rng = rand::thread_rng();
                    (0..fs_size).map(|_| T::rand(&mut rng)).collect()
                };
                // TODO: actually handle errors
                self.mpc
                    .input(mpc_sid, i, Some(alpha.clone() * &self.lp_coeff))
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input alpha for {i} {mpc_sid}"))
                    })?;
                self.mpc
                    .input_multi(mpc_sid, i, fs_size, Some(all_fs))
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input polys for {i} {mpc_sid}"))
                    })?;
            } else {
                self.mpc
                    .input::<T>(mpc_sid, i, None)
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input alpha for {i} {mpc_sid}"))
                    })?;
                self.mpc
                    .input_multi::<Vec<T>>(mpc_sid, i, fs_size, None)
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input polys for {i} {mpc_sid}"))
                    })?;
            }
        }

        let circuit: TCircuit<(I, Vec<(T, Vec<T>)>), Vec<Vec<(T, T)>>> = setup_circuit(
            self.n.into(),
            in_size,
            self.t.into(),
            &self.party_points,
            circuit,
        );

        let num_outputs = circuit.outputs.len();
        let party_outs = T::BIT_SIZE * num_shares * 2;
        let output_assignment = (0..num_outputs)
            .map(|w| (w, 1 + (w / party_outs) as i32))
            .collect();
        let parse_fn =
            |bits: &[bool]| <Vec<(T, T)> as CircuitCollection>::from_bits(num_shares, bits);

        trace!("running mpc eval on setup circuit");

        let mut output: Vec<(T, T)> = self
            .mpc
            .eval_generic(
                mpc_sid,
                &self.parties,
                &circuit,
                output_assignment,
                parse_fn,
            )
            .await?;

        let sr = self.state.as_mut().unwrap();
        let old_len = sr.share_inputs.len();
        sr.share_inputs.append(&mut output);

        let fsize = T::BIT_SIZE;
        for (i, &id) in out_ids.iter().enumerate() {
            sr.input_share_idx
                .insert(id, (old_len + (i / fsize), i % fsize));
        }

        Ok(())
    }

    async fn sample(&mut self, ids: &[InputId]) -> Result<(), UnexpectedError> {
        if self.state.is_none() {
            return Err(anyhow!("Init/resume not run").into());
        }
        let alpha = self.state.as_ref().unwrap().share_alpha.clone();

        let in_size = ids.len();

        let sid = SessionId::new(FuncId::Fthresh);
        let mpc_sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let mpc_sid = SessionId {
            parent: FuncId::Fthresh,
            id: mpc_sid,
        };

        self.mpc
            .init(mpc_sid, None)
            .await
            .with_context(|| self.err(sid, format!("Failed to initialize Fmpc {mpc_sid}")))?;

        let num_shares = (in_size + T::BIT_SIZE - 1) / T::BIT_SIZE;
        let fs_size: usize = 2 * num_shares * (self.t as usize - 1);

        for &i in self.parties.iter() {
            if i == self.party_id {
                let (all_fs, x_is): (Vec<_>, Vec<_>) = {
                    let mut rng = rand::thread_rng();
                    // make 2*l random degree t-1 polynomials to use as zero shares
                    let all_fs: Vec<T> = {
                        let mut rng = rand::thread_rng();
                        (0..fs_size).map(|_| T::rand(&mut rng)).collect()
                    };
                    let mut x_is = vec![false; in_size];
                    rng.fill(&mut x_is[..]);
                    (all_fs, x_is)
                };

                self.mpc
                    .input(mpc_sid, i, Some(alpha.clone() * &self.lp_coeff))
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input alpha for {i} {mpc_sid}"))
                    })?;

                self.mpc
                    .input_multi(mpc_sid, i, fs_size, Some(all_fs))
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input polys for {i} {mpc_sid}"))
                    })?;

                self.mpc
                    .input_multi(mpc_sid, i, in_size, Some(x_is))
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input xs for {i} {mpc_sid}"))
                    })?;
            } else {
                self.mpc
                    .input::<T>(mpc_sid, i, None)
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input alpha for {i} {mpc_sid}"))
                    })?;
                self.mpc
                    .input_multi::<Vec<T>>(mpc_sid, i, fs_size, None)
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input fs for {i} {mpc_sid}"))
                    })?;
                self.mpc
                    .input_multi::<Vec<bool>>(mpc_sid, i, in_size, None)
                    .await
                    .with_context(|| {
                        self.err(sid, format!("Failed to input xs for {i} {mpc_sid}"))
                    })?;
            }
        }

        let circuit: TCircuit<Vec<(T, Vec<T>, Vec<bool>)>, Vec<Vec<(T, T)>>> =
            sample_circuit(self.n.into(), in_size, self.t.into(), &self.party_points);

        let num_outputs = circuit.outputs.len();
        let party_outs = T::BIT_SIZE * num_shares * 2;
        let output_assignment = (0..num_outputs)
            .map(|w| (w, 1 + (w / party_outs) as i32))
            .collect();
        let parse_fn =
            |bits: &[bool]| <Vec<(T, T)> as CircuitCollection>::from_bits(num_shares, bits);
        let mut output: Vec<(T, T)> = self
            .mpc
            .eval_generic(
                mpc_sid,
                &self.parties,
                &circuit,
                output_assignment,
                parse_fn,
            )
            .await?;

        let sr = self.state.as_mut().unwrap();

        let old_len = sr.share_inputs.len();
        sr.share_inputs.append(&mut output);

        let fsize = T::BIT_SIZE;
        for (i, &id) in ids.iter().enumerate() {
            sr.input_share_idx
                .insert(id, (old_len + (i / fsize), i % fsize));
        }

        Ok(())
    }

    async fn eval<I, O: CircuitElement>(
        &mut self,
        parties: &[PartyId],
        ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<O, UnexpectedError> {
        assert_eq!(ids.len(), circuit.inputs.len());

        if self.state.is_none() {
            return Err(anyhow!("Init/resume not run").into());
        }
        // If we do not have the threshold of parties, abort
        if parties.len() < self.t.into() {
            return Err(anyhow!("Insufficient number of parties to run evaluation").into());
        }

        // If I am not part of the computation I should track that execution has occurred
        // but cannot return a value
        if !parties.contains(&self.party_id) {
            self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
            return Err(
                anyhow!("I am not one of the parties that is supposed to execute eval").into(),
            );
        }

        // calculate the lagrange coefficient for the given set of parties
        let party_points: Vec<_> = parties.iter().map(|&p| T::from(p.into())).collect();

        let lp = lagrange_poly(
            &party_points,
            &self.party_points[(self.party_id - 1) as usize],
            |x| *x,
        );

        let state = self.state.as_ref().unwrap();

        let alpha = state.share_alpha.clone();

        // accumulate the elements that contain the bits according to id
        // map of original index into share_inputs to index into xs/macs
        let mut found = HashMap::new();
        // the shares of x and ax
        let mut xs = Vec::new();
        let mut macs = Vec::new();
        // the list of bits to gather after validation
        let mut share_bits = Vec::new();
        let fsize = T::BIT_SIZE;

        for id in ids.iter() {
            let (idx, bit) = state.input_share_idx[id];
            if !found.contains_key(&idx) {
                let j = xs.len();
                found.insert(idx, j);
                let xy = &state.share_inputs[idx];
                xs.push(lp.clone() * xy.0);
                macs.push(lp.clone() * xy.1);
                share_bits.push(fsize * j + bit);
            } else {
                let si = found[&idx];
                share_bits.push(si * fsize + bit);
            }
        }

        let sid = self.cur_mpc_sid.fetch_add(1, Ordering::SeqCst);
        let sid = SessionId {
            parent: FuncId::Fthresh,
            id: sid,
        };

        let _ = self.mpc.init(sid, None).await?;

        for &i in parties.iter() {
            if i == self.party_id {
                let _ = self.mpc.input(sid, i, Some(alpha.clone() * &lp)).await;
                let _ = self
                    .mpc
                    .input_multi(sid, i, xs.len(), Some(xs.clone()))
                    .await;
                let _ = self
                    .mpc
                    .input_multi(sid, i, macs.len(), Some(macs.clone()))
                    .await;
            } else {
                let _ = self.mpc.input::<T>(sid, i, None).await;
                let _ = self.mpc.input_multi::<Vec<T>>(sid, i, xs.len(), None).await;
                let _ = self
                    .mpc
                    .input_multi::<Vec<T>>(sid, i, macs.len(), None)
                    .await;
            }
        }

        let augmented_circuit: TCircuit<Vec<(T, Vec<T>, Vec<T>)>, Option<O>> =
            add_validation_project(parties.len(), xs.len(), &share_bits, circuit);

        let output = self.mpc.eval_pub(sid, &parties, &augmented_circuit).await?;

        if output.is_none() {
            // cheat
            return Err(anyhow!("Cheat in input").into());
        }

        Ok(output.unwrap())
    }
}

impl<T: Field + RandElement + Copy, FM: AsyncMpc<T>> GenericThreshPlayer<T, FM> {
    pub fn new(party_id: PartyId, n: PartyId, t: u16, mpc: Arc<FM>) -> Result<Self, ()> {
        let parties: Vec<PartyId> = (1..n + 1).collect();
        let party_points: Vec<T> = (1..n + 1).map(|x| T::from(x.into())).collect();
        let lp = lagrange_poly(&party_points, &party_points[(party_id - 1) as usize], |x| {
            *x
        });

        Ok(GenericThreshPlayer {
            party_id: party_id,
            n: n,
            t: t,
            parties,
            party_points,
            lp_coeff: lp,
            mpc: mpc,
            state: None,
            cur_mpc_sid: AtomicU64::new(0),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        circuits::{aes::aes_key_schedule, arith::sum_circuit, new_builder, Gate},
        ff2_128::FF2_128,
        func_mpc::tests::build_test_mpcs,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };
    use tokio::{io, task::JoinSet};

    pub fn build_test_comcomps<FM: AsyncMpc<FF2_128>>(
        mpcs: &[Arc<FM>],
        t: PartyId,
    ) -> Vec<GenericThreshPlayer<FF2_128, FM>> {
        let num = mpcs.len();
        (1..=num)
            .map(|i| {
                GenericThreshPlayer::new(i as PartyId, num as PartyId, t, mpcs[i - 1].clone())
                    .unwrap()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_comcomp_execution() -> io::Result<()> {
        let party_info = get_test_party_infos(3);
        let mpcs = build_test_mpcs(&party_info);
        let comcomps = build_test_comcomps(&mpcs, 2);

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for mut cc in comcomps.into_iter() {
            js.spawn(async move {
                let _ = cc.init().await?;
                let ids: Vec<_> = (1..=256).collect();
                let _ = cc.sample(&ids).await?;
                let c = sum_circuit::<FF2_128>(2);
                cc.eval(&[1, 2, 3], &ids, &c).await
            });
        }

        let mut res = Vec::new();
        while let Some(x) = js.join_next().await {
            res.push(x.unwrap().unwrap());
        }

        // check that all parties got the same output
        assert_eq!(res[0], res[1]);
        assert_eq!(res[0], res[2]);

        Ok(())
    }

    fn example_setup_circuit<I: CircuitRing, O>(
        np: usize,
        cir: &TCircuit<I, O>,
    ) -> TCircuit<Vec<I>, (I, O)> {
        let b = new_builder();
        let (b, ids) = b.add_input_multi::<Vec<I>>(np, None);
        let sum = sum_circuit::<I>(np);
        let (b, k) = b.extend_circuit(&ids, &sum, None);
        let (b, mut keys) = b.extend_circuit(&k, cir, None);
        let mut outs = k; //vec![];//vec![k[0].clone()];
        outs.append(&mut keys);

        b.refine_input().refine_output(&outs).to_circuit()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_comcomp_setup() -> io::Result<()> {
        let party_info = get_test_party_infos(5);
        let mpcs = build_test_mpcs(&party_info);
        let comcomps = build_test_comcomps(&mpcs, 3);

        let aes = aes_key_schedule();

        let setup_c = example_setup_circuit(5, &aes);

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for mut cc in comcomps.into_iter() {
            let c = setup_c.clone();
            js.spawn(async move {
                let _ = cc.init().await?;
                let ids: Vec<_> = (0..c.outputs.len()).collect();
                let input = vec![false; 128];
                /*
                let c: TCircuit<[bool;5], [bool;129]> = {
                    let inputs: Vec<_> = (0..5).collect();
                    let mut gates = vec![
                        Gate::Xor(0,0,5),
                        Gate::Not(5,6),
                    ];
                    gates.extend((0..128).map(|i| Gate::And(i, i+1, 7+i)));

                    let outputs = (6..135).collect();

                    TCircuit::from_parts(inputs, gates, outputs)
                };
                */
                cc.setup(&input, &ids, &c).await
            });
        }

        let mut res = Vec::new();
        while let Some(x) = js.join_next().await {
            res.push(x.unwrap().unwrap());
        }

        // check that all parties got the same output
        assert_eq!(res[0], res[1]);
        assert_eq!(res[0], res[2]);

        Ok(())
    }
}
