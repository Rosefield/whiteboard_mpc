use crate::circuits::{
    circuit::{Circuit, Gate, TCircuit},
    elements::{CircuitCollection, CircuitElement},
    WireId,
};

use std::{collections::HashMap, marker::PhantomData};

/// CircuitBuilder can be used to iteratively build circuits, exposing ways to
/// refer to and extend subsections of the circuit.
pub struct CircuitBuilder<I, O> {
    _i: PhantomData<I>,
    _o: PhantomData<O>,
    inputs: Vec<WireId>,
    outputs: Vec<WireId>,
    consts: Option<[WireId; 2]>,
    next_id: WireId,
    named_sets: HashMap<&'static str, Vec<WireId>>,
    circuits: Vec<(Vec<WireId>, Circuit, Vec<WireId>)>,
}

impl<I, O> CircuitBuilder<I, O> {
    /// If added to the circuit, get the ids of the constants [0,1]
    pub fn get_const_wire_ids(&self) -> Option<[WireId; 2]> {
        self.consts.clone()
    }

    /// Add the constants [0,1] to the circuit
    pub fn with_consts(mut self) -> Self {
        let const_c: TCircuit<bool, [bool; 2]> = {
            let inputs = vec![0];
            let gates = vec![Gate::Xor(0, 0, 1), Gate::Not(1, 2)];
            let outputs = vec![1, 2];
            TCircuit::from_parts(inputs, gates, outputs)
        };

        let new_wires: Vec<_> = (self.next_id + 1..self.next_id + 3).collect();
        self.next_id += 3;

        self.consts = Some([new_wires[0], new_wires[1]]);
        self.circuits
            .push((vec![0], const_c.type_erase(), new_wires));

        self
    }

    pub fn encode_consts(&self, bits: &[bool]) -> Vec<WireId> {
        assert!(self.consts.is_some());

        let cs = self.consts.clone().unwrap();

        bits.iter()
            .map(|&b| if b { cs[1] } else { cs[0] })
            .collect()
    }

    /// Add an element as an input to the circuit
    /// Returning the new builder and the wire ids of the input
    pub fn add_input<T: CircuitElement>(
        self,
        name: Option<&'static str>,
    ) -> (CircuitBuilder<(I, T), O>, Vec<WireId>) {
        self.add_input_inner(name, T::BIT_SIZE)
    }

    /// Add `num` elements as inputs to the circuit
    /// Returning the new builder and the wire ids of the input
    pub fn add_input_multi<T: CircuitCollection>(
        self,
        num: usize,
        name: Option<&'static str>,
    ) -> (CircuitBuilder<(I, T), O>, Vec<WireId>) {
        self.add_input_inner(name, T::total_size(num))
    }

    fn add_input_inner<T>(
        self,
        name: Option<&'static str>,
        ts: usize,
    ) -> (CircuitBuilder<(I, T), O>, Vec<WireId>) {
        let mut i = self.inputs;
        let new_wires: Vec<WireId> = (self.next_id..self.next_id + ts).collect();

        i.extend(new_wires.iter());

        let mut ns = self.named_sets;
        if let Some(n) = name {
            ns.entry(n).or_insert(new_wires.clone());
        }

        let next_builder = CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: i,
            outputs: self.outputs,
            consts: self.consts,
            next_id: self.next_id + ts,
            named_sets: ns,
            circuits: self.circuits,
        };

        (next_builder, new_wires)
    }

    /// Get a set of wire ids that were previously registered under `name`
    pub fn get_named_ids(&self, name: &'static str) -> Vec<WireId> {
        self.named_sets[&name].clone()
    }

    /// Add a new circuit `c` to the builder, taking `ids` as the list of inputs
    /// to use as the input for `c`.
    /// Returns the new builder and the list of ids that are the output of `c`.
    pub fn extend_circuit<I2, O2>(
        self,
        ids: &[WireId],
        c: &TCircuit<I2, O2>,
        name: Option<&'static str>,
    ) -> (Self, Vec<WireId>) {
        assert_eq!(ids.len(), c.inputs.len());

        // map internal input id to external id
        let ci_remap: HashMap<WireId, WireId> =
            c.inputs.iter().cloned().zip(ids.iter().cloned()).collect();

        let mut next_id = self.next_id;
        // handle any passthrough wires, if the internal circuit just forwards an input
        // want to ensure we forward as well.
        let new_wires: Vec<WireId> = c
            .outputs
            .iter()
            .map(|co| {
                if ci_remap.contains_key(co) {
                    ci_remap[co]
                } else {
                    let new = next_id;
                    next_id += 1;
                    new
                }
            })
            .collect();

        let mut ns = self.named_sets;
        if let Some(n) = name {
            ns.entry(n).or_insert(new_wires.clone());
        }

        let mut circuits = self.circuits;
        let cc: TCircuit<_, _> = c.clone();
        let ct = cc.type_erase();
        circuits.push((ids.to_vec(), ct, new_wires.clone()));

        let next_builder = CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs,
            outputs: self.outputs,
            consts: self.consts,
            next_id: next_id,
            named_sets: ns,
            circuits: circuits,
        };

        (next_builder, new_wires)
    }

    /// Apply the circuit `c` to each of the `num` inputs specified by `ids`.
    /// Use `in_fn` to produce the list of ids for the circuit `c` to be able to
    /// be able to express having a closure for the map function.
    pub fn map_circuit<I2, T, F: Fn(&[WireId]) -> Vec<WireId>>(
        self,
        ids: &[WireId],
        c: &TCircuit<I2, T>,
        num: usize,
        in_fn: F,
        name: Option<&'static str>,
    ) -> (CircuitBuilder<I, (O, Vec<T>)>, Vec<WireId>) {
        assert!(ids.len() % num == 0);
        assert!(num >= 1);
        let i_size = ids.len() / num;
        let o_size = c.outputs.len();

        let mut circuits = self.circuits;
        let ol = num * c.outputs.len();
        let new_wires: Vec<WireId> = (self.next_id..self.next_id + ol).collect();

        let cc: TCircuit<_, _> = c.clone();
        let ct = cc.type_erase();
        //
        ids.chunks_exact(i_size)
            .zip(new_wires.as_slice().chunks_exact(o_size))
            .for_each(|(x, y)| {
                let input = in_fn(x);
                assert_eq!(input.len(), c.inputs.len());
                circuits.push((input, ct.clone(), y.to_vec()));
            });

        let mut ns = self.named_sets;
        if let Some(n) = name {
            ns.entry(n).or_insert(new_wires.clone());
        }

        let next_builder = CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs,
            outputs: self.outputs,
            consts: self.consts,
            next_id: self.next_id + ol,
            named_sets: ns,
            circuits: circuits,
        };

        (next_builder, new_wires)
    }

    /// Use the circuit `c` to reduce an array of `num` input elements to one.
    /// - `num` must be >= 2
    pub fn fold_circuit<T: CircuitElement>(
        self,
        ids: &[WireId],
        c: &TCircuit<(T, T), T>,
        num: usize,
        name: Option<&'static str>,
    ) -> (CircuitBuilder<I, (O, T)>, Vec<WireId>) {
        let s: usize = T::BIT_SIZE;
        assert!(ids.len() == s * num);
        assert!(num >= 2);
        assert!(c.outputs.len() == s);

        let mut circuits = self.circuits;
        let ol = (num - 1) * s;
        let new_wires: Vec<WireId> = (self.next_id..self.next_id + ol).collect();

        let cc: TCircuit<_, _> = c.clone();
        let ct = cc.type_erase();
        // reduce the first two elements
        circuits.push((ids[..2 * s].to_vec(), ct.clone(), new_wires[..s].to_vec()));

        // continue reducing with (in[i+2], out[i]) -> out[i+1]
        ids.chunks_exact(s)
            .skip(2)
            .zip(new_wires.as_slice().chunks_exact(s))
            .zip(new_wires.as_slice().chunks_exact(s).skip(1))
            .for_each(|((x, y), z)| {
                let mut input = x.to_vec();
                input.extend(y.iter());
                circuits.push((input, ct.clone(), z.to_vec()));
            });

        let final_wires = new_wires[ol - s..].to_vec();
        let mut ns = self.named_sets;
        if let Some(n) = name {
            ns.entry(n).or_insert(final_wires.clone());
        }

        let next_builder = CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs,
            outputs: self.outputs,
            consts: self.consts,
            next_id: self.next_id + ol,
            named_sets: ns,
            circuits: circuits,
        };

        (next_builder, final_wires)
    }

    /// Cast the input type of the current builder as `I2`.
    /// Useful to convert from the list representation to a single type
    /// e.g. start might be (((), T), T) and I2 = (T, T)
    pub fn refine_input<I2>(self) -> CircuitBuilder<I2, O> {
        CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs,
            outputs: self.outputs,
            consts: self.consts,
            next_id: self.next_id,
            named_sets: self.named_sets,
            circuits: self.circuits,
        }
    }

    /// Set the output of the circuit builder to be `output_ids` with type `O2`
    pub fn refine_output<O2>(self, output_ids: &[WireId]) -> CircuitBuilder<I, O2> {
        CircuitBuilder {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs,
            outputs: output_ids.to_vec(),
            consts: self.consts,
            next_id: self.next_id,
            named_sets: self.named_sets,
            circuits: self.circuits,
        }
    }

    /// Create a single typed circuit from this builder.
    pub fn to_circuit(&self) -> TCircuit<I, O> {
        // A circuit builder is a sequence of intermediary circuits
        // where the internal wires of those circuits are not immediately exposed to the builder.
        // The important part is that the circuits are self-contained, so only the input/output
        // wireids need to be preserved/mapped, and any internal wires can get new ids

        let mut gates = Vec::new();
        assert!(self.inputs.len() > 0 && self.outputs.len() > 0 && self.circuits.len() > 0);

        // we have a total number of wires inputs.len()+gates.len()
        let total_gates: usize = self.circuits.iter().map(|(_, c, _)| c.gates.len()).sum();
        let out_len = self.outputs.len();
        let total_wires = self.inputs.len() + total_gates;

        let mut id_remap: HashMap<WireId, WireId> = HashMap::new();

        let mut next_id = 0;
        let mut map_or_new = |map: &mut HashMap<_, _>, n: WireId| {
            if map.contains_key(&n) {
                map[&n]
            } else {
                let new = next_id;
                next_id += 1;
                map.insert(n, new);
                new
            }
        };
        // want inputs to be the first wires
        let inputs: Vec<_> = self
            .inputs
            .iter()
            .map(|&old| map_or_new(&mut id_remap, old))
            .collect();

        // for BF compatibility we need to make the outputs the last outputs.len() wire ids
        id_remap.extend(
            (total_wires - out_len..total_wires)
                .zip(self.outputs.iter())
                .map(|(new, old)| (*old, new)),
        );

        // TODO: this doesn't work if an input is also an output
        // It is OK if internal circuits have this property,
        assert!(
            id_remap.len() == self.inputs.len() + self.outputs.len(),
            "Builder inputs and outputs cannot overlap"
        );

        if let Some(_) = self.consts {
            // if we have constants we want to make sure we have a valid reference
            // to the first input wire
            id_remap.insert(0, 0);
        }

        let outputs = self.outputs.iter().map(|o| id_remap[o]).collect();

        for (c_in, c, c_out) in self.circuits.iter() {
            // map ids in c to ids for the builder
            let mut c_remap: HashMap<WireId, WireId> = c_in
                .iter()
                .zip(c.inputs.iter())
                .map(|(i, ci)| (*ci, id_remap[i]))
                .collect();

            // want to align c outputs to the ids in c_out
            c_remap.extend(c_out.iter().zip(c.outputs.iter()).map(|(o, co)| {
                let new = map_or_new(&mut id_remap, *o);
                (*co, new)
            }));

            let new_gates = c.gates.iter().map(|g| match g {
                Gate::And(x, y, z) => {
                    Gate::And(c_remap[x], c_remap[y], map_or_new(&mut c_remap, *z))
                }
                Gate::Xor(x, y, z) => {
                    Gate::Xor(c_remap[x], c_remap[y], map_or_new(&mut c_remap, *z))
                }
                Gate::Not(x, y) => Gate::Not(c_remap[x], map_or_new(&mut c_remap, *y)),
            });

            gates.extend(new_gates);
            assert_eq!(c_remap.len(), c_in.len() + c.gates.len());
        }

        // we should have assigned exactly the right number of wires dynamically.
        assert_eq!(next_id, total_wires - out_len);

        TCircuit {
            _i: PhantomData,
            _o: PhantomData,
            inputs,
            gates,
            outputs,
        }
    }
}

impl<I: CircuitElement, O: CircuitElement> CircuitBuilder<I, O> {
    /// Convert the builder to a circuit, and since we have restrictions on the
    /// types of the input/output check that the result is well-formed
    pub fn to_circuit_checked(&self) -> TCircuit<I, O> {
        self.to_circuit().checked()
    }
}

/// Create a new empty builder
pub fn new_builder() -> CircuitBuilder<(), ()> {
    CircuitBuilder {
        _i: PhantomData,
        _o: PhantomData,
        inputs: Vec::new(),
        outputs: Vec::new(),
        consts: None,
        next_id: 0,
        named_sets: HashMap::new(),
        circuits: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_circuit() {
        let builder = new_builder();

        let (builder, in1) = builder.add_input::<[bool; 4]>(Some("in1"));
        let (builder, in2) = builder.add_input::<[bool; 4]>(Some("in2"));
        let builder = builder.refine_input::<([bool; 4], [bool; 4])>();

        let c: TCircuit<([bool; 4], [bool; 4]), [bool; 4]> = {
            let inputs = (0..8).collect();
            let gates = (0..4).map(|l| Gate::Xor(l, l + 4, l + 8)).collect();
            let outputs = (8..12).collect();

            TCircuit::from_parts(inputs, gates, outputs)
        };

        let mut ins = in1;
        ins.extend(in2.iter());

        let (builder, res) = builder.extend_circuit(&ins, &c, None);
        let mut ins2 = res;
        ins2.extend(in2);
        let (builder, res2) = builder.extend_circuit(&ins2, &c, None);
        let builder = builder.refine_output::<[bool; 4]>(&res2);

        let new_c = builder.to_circuit_checked();
        assert!(new_c.inputs.len() == 8);
        assert!(new_c.outputs.len() == 4);
        assert!(new_c.gates.len() == 8);
    }
}
