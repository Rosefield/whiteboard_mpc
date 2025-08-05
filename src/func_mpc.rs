use crate::{
    auth_bits::{Abit, Abits},
    base_func::{BaseFunc, FuncId, SessionId, UnexpectedError},
    circuits::{CircuitCollection, CircuitElement, TCircuit},
    ffi::ffi::{
        make_bristol_circuit, make_network, run_mpc, Abit as FFI_Abit, MpcOut,
        Network as FFI_Network, PartyInfo as FFI_Party,
    },
    field::Field,
    party::{PartyId, PartyInfo},
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use anyhow::Context;

use log::{info, trace};

#[derive(Debug)]
pub struct WrkMpcPlayer<T> {
    party_id: PartyId,
    //n: u16,
    party_info: Vec<PartyInfo>,
    net: Arc<OnceLock<cxx::SharedPtr<FFI_Network>>>,
    run_infos: Mutex<HashMap<SessionId, RunInfo<T>>>,
    execution_stats: Mutex<HashMap<SessionId, Stats>>,
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub time: Duration,
    pub network_bytes: u64,
}

#[derive(Debug)]
struct RunInfo<T> {
    num_input_wires: usize,
    my_input: Vec<bool>,
    delta: Option<T>,
    auth_inputs: Option<Abits<T>>,
    input_assignment: HashMap<usize, i32>,
}

impl<T> RunInfo<T> {
    fn new(delta: Option<T>) -> Self {
        RunInfo {
            num_input_wires: 0,
            my_input: Vec::new(),
            delta: delta,
            auth_inputs: None,
            input_assignment: HashMap::new(),
        }
    }
}

pub type InputId = u16;

impl<T> BaseFunc for WrkMpcPlayer<T> {
    const FUNC_ID: FuncId = FuncId::Fmpc;
    const REQUIRED_FUNCS: &'static [FuncId] = &[];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// Trait to represent boolean-circuit F_MPC
pub trait AsyncMpc<TF> {
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId, delta: Option<TF>) -> Result<(), UnexpectedError>;

    /// Add an input from `party` to the `sid` instance
    async fn input<T: CircuitElement>(
        &self,
        sid: SessionId,
        party: PartyId,
        input: Option<T>,
    ) -> Result<(), UnexpectedError>;

    /// Add `num` inputs from `party` to the `sid` instance
    async fn input_multi<T: CircuitCollection>(
        &self,
        sid: SessionId,
        party: PartyId,
        num: usize,
        input: Option<T>,
    ) -> Result<(), UnexpectedError>;

    /// Add `num` inputs from `party` to the `sid` instance
    async fn input_abit(&self, sid: SessionId, input: Abits<TF>) -> Result<(), UnexpectedError>;

    /// Evaluate `circuit` using the inputs that were previously supplied
    /// giving the output of the circuit to all parties
    async fn eval_pub<I, O: CircuitElement>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
    ) -> Result<O, UnexpectedError>;

    /// Evaluate `circuit` using the inputs that were previously supplied
    /// giving the i-th output of the circuit to party i.
    async fn eval_priv<I, O: CircuitElement>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, Vec<O>>,
    ) -> Result<O, UnexpectedError>;

    /// Evaluate `circuit` using the inputs that were previously supplied
    /// giving authenticated output shares
    async fn eval_abit<I, O: CircuitElement>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
    ) -> Result<Abits<TF>, UnexpectedError>;

    /// Evaluate `circuit` using the inputs that were previously supplied
    /// giving output wires to parties according to `output_assignment`
    /// and parsing the output with `parse_fn`
    async fn eval_generic<I, O, O2, F: FnOnce(&[bool]) -> O2>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
        output_assignment: HashMap<usize, i32>,
        parse_fn: F,
    ) -> Result<O2, UnexpectedError>;
}

/// Currently just a thin-wrapper around the mpc_runner executable.
/// Collects inputs from parties, and marshals the data to/from mpc_runner
impl<TF: Field + CircuitElement + 'static> AsyncMpc<TF> for WrkMpcPlayer<TF> {
    async fn init(&self, sid: SessionId, delta: Option<TF>) -> Result<(), UnexpectedError> {
        {
            let mut ri = self.run_infos.lock().unwrap();
            assert!(!ri.contains_key(&sid));
            ri.insert(sid, RunInfo::new(delta));
        }

        let party_info: Vec<_> = self
            .party_info
            .iter()
            .map(|p| FFI_Party {
                id: p.id,
                ip: p.ip.to_string(),
                port: p.port,
            })
            .collect();

        let net_lock = self.net.clone();
        let my_id = self.party_id;
        let _ = tokio::task::spawn_blocking(move || {
            let _ = net_lock.get_or_init(|| make_network(my_id, &party_info, 100).unwrap());
        })
        .await;

        Ok(())
    }

    async fn input<T: CircuitElement>(
        &self,
        sid: SessionId,
        party: PartyId,
        input: Option<T>,
    ) -> Result<(), UnexpectedError> {
        // Add each parties' inputs,
        let mut ris = self.run_infos.lock().unwrap();
        assert!(ris.contains_key(&sid));
        ris.entry(sid).and_modify(|ri| {
            let next_id = ri.num_input_wires;
            ri.num_input_wires += T::BIT_SIZE;

            // mark the next T::BIT_SIZE wires as being from party
            ri.input_assignment
                .extend((next_id..next_id + T::BIT_SIZE).map(|l| (l, party as i32)));

            if self.party_id == party {
                // If I am the specified party, I should be providing an input
                assert!(input.is_some());
                let prev_len = ri.my_input.len();
                ri.my_input.resize(prev_len + T::BIT_SIZE, false);
                let i = input.unwrap();
                i.to_bits(&mut ri.my_input.as_mut_slice()[prev_len..]);
            }
        });
        Ok(())
    }

    async fn input_multi<T: CircuitCollection>(
        &self,
        sid: SessionId,
        party: PartyId,
        num: usize,
        input: Option<T>,
    ) -> Result<(), UnexpectedError> {
        // Add each parties' inputs,
        let mut ris = self.run_infos.lock().unwrap();
        assert!(ris.contains_key(&sid));
        ris.entry(sid).and_modify(|ri| {
            let num_bits = T::total_size(num);
            let next_id = ri.num_input_wires;
            ri.num_input_wires += num_bits;

            // mark the next T::BIT_SIZE wires as being from party
            ri.input_assignment
                .extend((next_id..next_id + num_bits).map(|l| (l, party as i32)));

            if self.party_id == party {
                // If I am the specified party, I should be providing an input
                assert!(input.is_some());
                let prev_len = ri.my_input.len();
                ri.my_input.resize(prev_len + num_bits, false);
                let i = input.unwrap();
                i.to_bits(&mut ri.my_input.as_mut_slice()[prev_len..]);
            }
        });
        Ok(())
    }

    async fn input_abit(&self, sid: SessionId, input: Abits<TF>) -> Result<(), UnexpectedError> {
        let mut ris = self.run_infos.lock().unwrap();
        assert!(ris.contains_key(&sid));
        ris.entry(sid).and_modify(|ri| {
            let num_bits = input.len();
            let next_id = ri.num_input_wires;
            ri.num_input_wires += num_bits;

            // assign each of the input bits as auth input
            ri.input_assignment
                .extend((next_id..next_id + num_bits).map(|l| (l, -1)));

            if let Some(ai) = ri.auth_inputs.as_mut() {
                ai.append(input);
            } else {
                ri.auth_inputs = Some(input);
            }
        });
        Ok(())
    }

    async fn eval_pub<I, O: CircuitElement>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
    ) -> Result<O, UnexpectedError> {
        // "0" party assignment is public output
        let output_assignment = (0..O::BIT_SIZE).map(|w| (w, 0)).collect();
        let (out, _) = self
            .eval_inner(sid, parties, output_assignment, circuit, O::from_bits)
            .await?;
        Ok(out)
    }

    async fn eval_priv<I, O: CircuitElement>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, Vec<O>>,
    ) -> Result<O, UnexpectedError> {
        // There should be enough wires for one output per party
        assert_eq!(
            circuit.outputs.len(),
            <Vec::<O> as CircuitCollection>::total_size(parties.len())
        );

        // Assign the O elements to parties 1,2,... in order
        let output_assignment = (0..circuit.outputs.len())
            .map(|w| (w, parties[w / O::BIT_SIZE] as i32))
            .collect();

        let (out, _) = self
            .eval_inner(sid, parties, output_assignment, circuit, O::from_bits)
            .await?;
        Ok(out)
    }

    async fn eval_abit<I, O>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
    ) -> Result<Abits<TF>, UnexpectedError> {
        // Assign the O elements to be authenticated output
        let output_assignment = (0..circuit.outputs.len()).map(|w| (w, -1)).collect();

        let (_, abits) = self
            .eval_inner(sid, parties, output_assignment, circuit, |_| ())
            .await?;

        Ok(abits)
    }

    async fn eval_generic<I, O, O2, F: FnOnce(&[bool]) -> O2>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        circuit: &TCircuit<I, O>,
        output_assignment: HashMap<usize, i32>,
        parse_fn: F,
    ) -> Result<O2, UnexpectedError> {
        // there should be assignments for each wire
        assert!(circuit.outputs.len() == output_assignment.len());

        let (out, _) = self
            .eval_inner(sid, parties, output_assignment, circuit, parse_fn)
            .await?;
        Ok(out)
    }
}

impl<T> WrkMpcPlayer<T> {
    pub fn new(party_id: PartyId, party_info: &[PartyInfo]) -> Result<Self, ()> {
        Ok(WrkMpcPlayer {
            party_id: party_id,
            party_info: party_info.to_vec(),
            net: Arc::new(OnceLock::new()),
            run_infos: Mutex::new(HashMap::new()),
            execution_stats: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_execution_stats(&self) -> HashMap<SessionId, Stats> {
        self.execution_stats.lock().unwrap().clone()
    }
}

impl<T: Field + CircuitElement + 'static> WrkMpcPlayer<T> {
    async fn eval_inner<I, O, O2, F: FnOnce(&[bool]) -> O>(
        &self,
        sid: SessionId,
        parties: &[PartyId],
        output_assignment: HashMap<usize, i32>,
        circuit: &TCircuit<I, O2>,
        parse_fn: F,
    ) -> Result<(O, Abits<T>), UnexpectedError> {
        let np = parties.len();

        let n_out = output_assignment.len();
        let output_assignment = (0..n_out).map(|i| output_assignment[&i]).collect();

        let (delta, input_assignment, my_input, auth_input) = {
            let ris = self.run_infos.lock().unwrap();
            assert!(ris.contains_key(&sid));

            let ri = &ris[&sid];
            // make sure the circuit has the same number of input wires as we have received inputs
            assert_eq!(circuit.inputs.len(), ri.num_input_wires);

            let input_assignment = (0..ri.num_input_wires)
                .map(|i| ri.input_assignment[&i] as i32)
                .collect();

            let auth_input: Vec<Abit<T>> = if let Some(ai) = ri.auth_inputs.as_ref() {
                ai.into()
            } else {
                Vec::new()
            };

            (
                ri.delta.clone(),
                input_assignment,
                ri.my_input.clone(),
                auth_input,
            )
        };

        let bf = make_bristol_circuit(
            circuit.inputs.len() as i32,
            circuit.outputs.len() as i32,
            circuit.make_ffi_gates(),
        );

        let start = Instant::now();

        let my_id = self.party_id;
        info!(
            "{}: sid {} evaluating circuit of size {}",
            my_id,
            sid,
            circuit.describe()
        );

        let net = self.net.get().unwrap().clone();

        let pids = parties.to_vec();

        let h = tokio::task::spawn_blocking(move || {
            let ai: Vec<Abit<T>> = auth_input;
            let ffi_ai = ai
                .iter()
                .map(|a| FFI_Abit {
                    bit: a.bit,
                    macs: unsafe { std::mem::transmute(a.macs.as_slice()) },
                    keys: unsafe { std::mem::transmute(a.keys.as_slice()) },
                })
                .collect();
            let delta: Option<T> = delta;
            let delta_b = {
                if let Some(d) = delta {
                    let mut delta_b = vec![false; T::BIT_SIZE];
                    d.to_bits(&mut delta_b);
                    delta_b
                } else {
                    Vec::new()
                }
            };

            trace!("{}: sid {} starting mpc", my_id, sid);

            run_mpc(
                my_id,
                &pids,
                net,
                delta_b,
                input_assignment,
                my_input,
                ffi_ai,
                output_assignment,
                bf,
            )
        });

        let MpcOut {
            outs: bits,
            auth_outs: abit_bytes,
            bytes_sent: network_bytes,
        } = h
            .await
            .unwrap()
            .with_context(|| self.err(sid, "Failed to run the MPC"))?;

        let stats = Stats {
            time: start.elapsed(),
            network_bytes: network_bytes,
        };

        info!("{}: sid {} execution stats {:?}", self.party_id, sid, stats);

        {
            let mut g = self.execution_stats.lock().unwrap();
            g.insert(sid, stats);
        }

        {
            let mut ris = self.run_infos.lock().unwrap();
            ris.remove(&sid);
        }

        let mut abits = Abits::empty(np - 1);

        // the return value has enough bytes for all parties, but we of course
        // don't care about anything that is at our index
        abit_bytes.chunks_exact(1 + 32 * np).for_each(|c| {
            abits.bits.push(c[0] == 1);

            for (i, &j) in parties.iter().filter(|&p| *p != self.party_id).enumerate() {
                let j = (j - 1) as usize;
                abits.keys[i].push(T::from_bytes(&c[1 + j * 16..1 + (j + 1) * 16]));
                abits.macs[i].push(T::from_bytes(&c[1 + (np + j) * 16..1 + (np + j + 1) * 16]));
            }
        });

        Ok((parse_fn(bits.as_slice()), abits))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        circuits::arith::sum_circuit,
        circuits::Gate,
        ff2_128::FF2_128,
        field::RandElement,
        func_net::tests::{build_test_nets, get_test_party_infos},
    };
    use std::sync::Arc;
    use tokio::task::JoinSet;

    pub fn build_test_mpcs(party_info: &[PartyInfo]) -> Vec<Arc<WrkMpcPlayer<FF2_128>>> {
        let num = party_info.len() as PartyId;
        (1..=num)
            .map(|i| Arc::new(WrkMpcPlayer::new(i, &party_info).unwrap()))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_pub_out() {
        let party_info = get_test_party_infos(3);
        let mpcs: Vec<_> = (1..4)
            .map(|i| (i, WrkMpcPlayer::<FF2_128>::new(i, &party_info).unwrap()))
            .collect();

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for (id, m) in mpcs.into_iter() {
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                let _ = m.init(sid, None).await?;
                for j in 1..4 {
                    let input = if j == id {
                        Some(FF2_128::new(0, 1 << (j - 1)))
                    } else {
                        None
                    };
                    let _ = m.input(sid, j, input).await?;
                }
                let sum_c = sum_circuit::<FF2_128>(3);
                let res = m.eval_pub(sid, &[1, 2, 3], &sum_c).await?;
                Ok(res)
            });
        }

        while let Some(r) = js.join_next().await {
            let s = r.unwrap().unwrap();
            assert_eq!(s, FF2_128::new(0, 7));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_abit_in_out() {
        let party_info = get_test_party_infos(3);
        let mpcs: Vec<_> = (1..4)
            .map(|i| (i, WrkMpcPlayer::<FF2_128>::new(i, &party_info).unwrap()))
            .collect();

        let deltas: Vec<_> = {
            let mut rng = rand::thread_rng();
            (0..3).map(|_| FF2_128::rand(&mut rng)).collect()
        };

        let mut js = JoinSet::<Result<_, UnexpectedError>>::new();
        for ((id, m), d) in mpcs.into_iter().zip(deltas.into_iter()) {
            js.spawn(async move {
                let sid = SessionId::new(FuncId::Ftest);
                m.init(sid, Some(d)).await?;
                for j in 1..4 {
                    let input = if j == id {
                        Some(FF2_128::new(0, 1 << (j - 1)))
                    } else {
                        None
                    };
                    let _ = m.input(sid, j, input).await?;
                }
                let sum_c = sum_circuit::<FF2_128>(3);
                let abits = m.eval_abit(sid, &[1, 2, 3], &sum_c).await?;

                let sid2 = sid.next();

                m.init(sid2, Some(d)).await?;
                m.input_abit(sid2, abits).await?;
                let c: TCircuit<FF2_128, FF2_128> = {
                    let inputs = (0..128).collect();
                    let outputs = (256..384).collect();
                    let mut gates: Vec<_> =
                        (0..128).map(|i| Gate::Xor(i, 127 - i, 128 + i)).collect();
                    gates.extend((0..128).map(|i| Gate::And(128 + i, 128 + i, 256 + i)));

                    TCircuit::from_parts(inputs, gates, outputs)
                };

                let res = m.eval_pub(sid2, &[1, 2, 3], &c).await?;
                Ok(res)
            });
        }

        while let Some(r) = js.join_next().await {
            let s = r.unwrap().unwrap();
            assert_eq!(s, FF2_128::new(7 << 61, 7));
        }
    }
}
