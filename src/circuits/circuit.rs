use crate::{
    circuits::{elements::CircuitElement, WireId},
    ffi::ffi::Gate as FFI_Gate,
};

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    marker::PhantomData,
    str::FromStr,
};

/// Represents an arbitrary boolean circuit
#[derive(Clone)]
pub struct Circuit {
    pub inputs: Vec<WireId>,
    pub gates: Vec<Gate>,
    pub outputs: Vec<WireId>,
}

/// Represents an arbitrary boolean circuit, with typed inputs and outputs
/// That is, `I` can convert to an array of input wires `inputs`,
/// and the array of output wires `outputs` can be parsed as `O`.
#[derive(PartialEq, Eq, Debug)]
pub struct TCircuit<I, O> {
    pub(super) _i: PhantomData<I>,
    pub(super) _o: PhantomData<O>,

    pub inputs: Vec<WireId>,
    pub gates: Vec<Gate>,
    pub outputs: Vec<WireId>,
}

impl<I, O> Clone for TCircuit<I, O> {
    fn clone(&self) -> Self {
        Self {
            _i: PhantomData,
            _o: PhantomData,
            inputs: self.inputs.clone(),
            gates: self.gates.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl<I, O> TCircuit<I, O> {
    /// Convert to a circuit without type information
    pub fn type_erase(self) -> Circuit {
        Circuit {
            inputs: self.inputs,
            gates: self.gates,
            outputs: self.outputs,
        }
    }

    pub fn describe(&self) -> String {
        let nands: usize = self
            .gates
            .iter()
            .map(|g| match g {
                Gate::And(_, _, _) => 1usize,
                _ => 0,
            })
            .sum();

        return format!(
            "(|I| = {}, |G| = {}, |O| = {})",
            self.inputs.len(),
            nands,
            self.outputs.len()
        );
    }

    pub fn make_ffi_gates(&self) -> Vec<FFI_Gate> {
        self.gates
            .iter()
            .map(|g| match g {
                Gate::And(a, b, c) => FFI_Gate {
                    in1: (*a) as i32,
                    in2: (*b) as i32,
                    out: (*c) as i32,
                    kind: 0,
                },
                Gate::Xor(a, b, c) => FFI_Gate {
                    in1: (*a) as i32,
                    in2: (*b) as i32,
                    out: (*c) as i32,
                    kind: 1,
                },
                Gate::Not(a, b) => FFI_Gate {
                    in1: (*a) as i32,
                    in2: 0,
                    out: (*b) as i32,
                    kind: 2,
                },
            })
            .collect()
    }

    /// Outputs the circuit as a Bristol Format file
    pub fn write_to_file_bf(&self, file_name: &str) {
        let mut f = BufWriter::new(File::create(file_name).unwrap());

        let num_wires = self.inputs.len() + self.gates.len();
        let num_outs = self.outputs.len();
        let expected_outs: HashSet<_> = (num_wires - num_outs..num_wires).collect();
        self.outputs
            .iter()
            .for_each(|o| assert!(expected_outs.contains(o)));

        writeln!(f, "{} {}", self.gates.len(), num_wires).unwrap();
        writeln!(f, "{} 0 {}", self.inputs.len(), self.outputs.len()).unwrap();
        writeln!(f, "").unwrap();

        for g in self.gates.iter() {
            match g {
                Gate::And(a, b, c) => {
                    writeln!(f, "2 1 {} {} {} AND", a, b, c).unwrap();
                }
                Gate::Xor(a, b, c) => {
                    writeln!(f, "2 1 {} {} {} XOR", a, b, c).unwrap();
                }
                Gate::Not(a, b) => {
                    writeln!(f, "1 1 {} {} INV", a, b).unwrap();
                }
            };
        }

        f.flush().unwrap();
    }

    /// Reads the circuit description from a (modified) Peralta-style file
    /// I don't know if there is a formal name for this syntax, but the source
    /// was Peralta's circuit-minimization website
    pub fn read_from_file_peralta(file_name: &str) -> Result<Self, ()> {
        let mut f = BufReader::new(File::open(file_name).unwrap());
        let mut buf = String::new();

        let read_first_num = |f: &mut BufReader<_>| {
            let mut b = String::new();
            let _ = f.read_line(&mut b);
            let n = b
                .split(' ')
                .map(|x| WireId::from_str(x).unwrap())
                .next()
                .unwrap();
            n
        };

        // First line is num_gates
        let ng = read_first_num(&mut f);

        // Second line is num_inputs
        let n_in = read_first_num(&mut f);

        let inputs = (0..n_in).collect();

        let mut idmap: HashMap<String, WireId> = HashMap::new();

        // third line is list of input ids
        let _ = f.read_line(&mut buf);
        buf.split(char::is_whitespace)
            .filter(|x| !x.trim().is_empty())
            .enumerate()
            .for_each(|(i, n)| {
                idmap.insert(n.to_string(), i);
            });
        buf.clear();

        // fourth line is number of outputs
        let n_out = read_first_num(&mut f);

        let mut outmap: HashMap<String, WireId> = HashMap::new();
        // fifth line is list of output ids
        let _ = f.read_line(&mut buf);
        let mut output_names: Vec<_> = buf
            .split(char::is_whitespace)
            .filter(|x| !x.trim().is_empty())
            .map(|n| n.to_string())
            .collect();
        assert!(n_out == output_names.len());
        // for whatever reason the list of output wires is not ordered by
        // their semantic value, i.e. the output list may be C0 C1 C2 C3 C7 C11 C15 C8 ...
        // where you would want to read the values in numerical order
        output_names.sort_by_key(|f| WireId::from_str(&f[1..]).unwrap());
        output_names.iter().enumerate().for_each(|(i, n)| {
            outmap.insert(n.to_string(), i);
        });
        buf.clear();

        // sixth line is "begin"
        let _ = f.read_line(&mut buf);
        buf.clear();

        let mut next_id = n_in;

        // remaining lines are the gates
        let gates: Vec<_> = f
            .lines()
            .filter_map(|l| {
                if l.is_err() {
                    return None;
                }
                let line = l.unwrap();

                let s: Vec<_> = line.split(char::is_whitespace).collect();
                // either a3 = a1 op a2
                // or o1 = a1
                if s.len() == 5 {
                    let id = next_id;
                    idmap.insert(s[0].to_string(), id);
                    next_id += 1;
                    match s[3] {
                        "+" => Some(Gate::Xor(idmap[s[2]], idmap[s[4]], id)),
                        "x" => Some(Gate::And(idmap[s[2]], idmap[s[4]], id)),
                        _ => None,
                    }
                } else {
                    outmap
                        .entry(s[0].to_string())
                        .and_modify(|v| *v = idmap[s[2]]);

                    None
                }
            })
            .collect();

        assert!(gates.len() == ng);

        let outputs = output_names.iter().map(|n| outmap[n]).collect();

        Ok(TCircuit {
            _i: PhantomData,
            _o: PhantomData,
            inputs,
            gates,
            outputs,
        })
    }

    /// Reads a Bristol-format circuit from `file_name`
    pub fn read_from_file_bf(file_name: &str) -> Result<Self, ()> {
        let mut f = BufReader::new(File::open(file_name).unwrap());

        let read_nums = |f: &mut BufReader<_>| {
            let mut b = String::new();
            let _ = f.read_line(&mut b);
            let ns = b
                .split(char::is_whitespace)
                .filter(|x| !x.trim().is_empty())
                .map(|x| WireId::from_str(x).unwrap())
                .collect::<Vec<_>>();
            ns
        };
        // First line is num_gates num_wires
        let ns = read_nums(&mut f);
        let (ng, nw) = (ns[0], ns[1]);

        // Second line is num_inputs_1 num_inputs_2 num_outputs
        let ns = read_nums(&mut f);
        let (n_in1, n_in2, n_out) = (ns[0], ns[1], ns[2]);
        assert_eq!(n_in1 + n_in2 + ng, nw);

        let inputs = (0..n_in1 + n_in2).collect();

        let mut gates = Vec::with_capacity(ng);

        let num = |n: &str| WireId::from_str(n).unwrap();

        for l in f.lines() {
            if l.is_err() {
                continue;
            }
            let line = l.unwrap();

            let s: Vec<_> = line.split(' ').collect();
            if let Some(g) = match *s.last().unwrap() {
                "AND" => Some(Gate::And(num(s[2]), num(s[3]), num(s[4]))),
                "XOR" => Some(Gate::Xor(num(s[2]), num(s[3]), num(s[4]))),
                "INV" => Some(Gate::Not(num(s[2]), num(s[3]))),
                _ => None,
            } {
                gates.push(g);
            }
        }

        assert_eq!(gates.len(), ng);

        let outputs = (nw - n_out..nw).collect();

        Ok(TCircuit {
            _i: PhantomData,
            _o: PhantomData,
            inputs,
            gates,
            outputs,
        })
    }

    pub fn well_formed(self) -> Self {
        // input/output ids are distinct
        let iset: HashSet<_> = self.inputs.iter().cloned().collect();
        assert!(self.inputs.len() == iset.len());
        let oset: HashSet<_> = self.outputs.iter().cloned().collect();
        assert!(self.outputs.len() == oset.len());
        //assert!(iset.intersection(&oset).collect::<HashSet<_>>().len() == 0);

        // each gate has a distinct output id
        let mut wset: HashSet<_> = self.gates.iter().map(|g| g.output_id()).collect();
        assert!(self.gates.len() == wset.len());

        // gates are in a topologically valid order
        let mut wires_so_far = iset.clone();
        self.gates.iter().for_each(|g| {
            let out = g.output_id();
            match g {
                Gate::And(x, y, _) => assert!(wires_so_far.contains(x) && wires_so_far.contains(y)),
                Gate::Xor(x, y, _) => assert!(wires_so_far.contains(x) && wires_so_far.contains(y)),
                Gate::Not(x, _) => assert!(wires_so_far.contains(x)),
            };
            wires_so_far.insert(out);
        });

        // gates do not overlap output id with the inputs
        wset.extend(iset.iter());
        assert!(wset.len() == self.gates.len() + self.inputs.len());

        // all output wires are in the wire set
        assert!(wset.intersection(&oset).collect::<HashSet<_>>().len() == oset.len());

        self
    }
}

impl<I: CircuitElement, O: CircuitElement> TCircuit<I, O> {
    /// Check that the circuit is well-formed:
    /// - Inputs and outputs are the appropriate size
    /// - Inputs and outputs have unique wire ids
    /// - Gates are in a topologically valid order
    /// - All gates have a unique wire id output
    /// - All output wire ids are in the set of {inputs, gates}
    pub fn checked(self) -> Self {
        // right number of inputs and outputs
        assert!(I::BIT_SIZE == self.inputs.len());
        assert!(O::BIT_SIZE == self.outputs.len());

        self.well_formed()
    }

    /// Makes a circuit from its parts, and check that it is well-formed
    pub fn from_parts(inputs: Vec<WireId>, gates: Vec<Gate>, outputs: Vec<WireId>) -> Self {
        TCircuit {
            _i: PhantomData,
            _o: PhantomData,
            inputs,
            gates,
            outputs,
        }
        .checked()
    }
}

/// Description of a gate in a circuit, as Op(inputs..., output_id)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    //ConstZero,
    //ConstOne,
    //Input(WireId),
    And(WireId, WireId, WireId),
    Xor(WireId, WireId, WireId),
    Not(WireId, WireId),
}

impl Gate {
    /// Get the id of the output for this gate
    pub fn output_id(&self) -> WireId {
        match self {
            Self::And(_, _, x) => *x,
            Self::Xor(_, _, x) => *x,
            Self::Not(_, x) => *x,
        }
    }
}
