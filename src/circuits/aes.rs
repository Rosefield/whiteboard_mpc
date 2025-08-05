use crate::circuits::{
    builder::new_builder,
    circuit::{Gate, TCircuit},
    get_def_circuit,
};

/// Generated the AES key schedule for AES-128, excluding the input key
/// Thus the full 1408 schedule is K || schedule(K)
pub fn aes_key_schedule() -> TCircuit<[bool; 128], [bool; 1280]> {
    let b = new_builder();
    let (mut b, i) = b.add_input::<[bool; 128]>(None);
    // key is a sequence of bits 0,1,...,127
    // or bytes a0,..,a15,

    // sbox, takes a big-endian byte, e.g. for bits 0,1,2,3,4,5,6,7 input xi is bit 7-i
    let sbox = aes_sbox();

    // first key is just the input
    let mut outputs = i;

    // rcon is 32-bit word rc || 0*24
    let rcs = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

    // derived key for 10 rounds, each W is 32-bits
    for i in 0..10 {
        // let r = 4(i+1)
        let last_block = &outputs[outputs.len() - 128..];
        // S(Rot(W_{r-1}))
        let (n, mut s0) = b.extend_circuit(&last_block[104..112], &sbox, None);
        let (n, mut s1) = n.extend_circuit(&last_block[112..120], &sbox, None);
        let (n, mut s2) = n.extend_circuit(&last_block[120..128], &sbox, None);
        let (n, mut s3) = n.extend_circuit(&last_block[96..104], &sbox, None);

        let rcircuit: TCircuit<[bool; 160], [bool; 128]> = {
            let inputs: Vec<_> = (0..160).collect();
            let wis = &inputs[..128];
            let si = &inputs[128..];

            let mut outputs: Vec<_> = (160..192).collect();

            let mut next = 192;

            //W_{r + 0} = W_{r-4} + S(Rot(W_{r-1})) + (rc_i << 24)
            let mut gates: Vec<_> = (0..32)
                .map(|j| Gate::Xor(wis[j], si[j], 160 + j))
                .chain((0..8).filter_map(|j| {
                    // xor constant => not if const bit is 1
                    // bytes are MSB first
                    if (rcs[i] >> (7 - j)) & 1 == 1 {
                        let g = Gate::Not(outputs[j], next);
                        outputs[j] = next;
                        next += 1;
                        Some(g)
                    } else {
                        None
                    }
                }))
                .collect();

            //W_{r + j} = W_{r + j - 4} + W_{r + j - 1}
            outputs.extend((0..96).map(|j| next + j));
            gates.extend((0..96).map(|j| Gate::Xor(inputs[32 + j], outputs[j], next + j)));

            TCircuit::from_parts(inputs, gates, outputs)
        };

        let mut ins = last_block.to_vec();
        ins.append(&mut s0);
        ins.append(&mut s1);
        ins.append(&mut s2);
        ins.append(&mut s3);

        let (n, mut ws) = n.extend_circuit(&ins, &rcircuit, None);
        outputs.append(&mut ws);
        b = n;
    }

    // because of BF compatibility requirements we cannot output an input wire
    // so just output the derived wires
    b.refine_input().refine_output(&outputs[128..]).to_circuit()
}

pub fn aes_with_schedule() -> TCircuit<([bool; 128], [bool; 1408]), [bool; 128]> {
    TCircuit::read_from_file_bf(&get_def_circuit("aes_128_expanded.txt")).unwrap()
}

/// Produces a circuit for the AES SBox,
/// transcribed from https://eprint.iacr.org/2009/191.pdf
pub fn aes_sbox() -> TCircuit<[bool; 8], [bool; 8]> {
    TCircuit::read_from_file_bf(&get_def_circuit("aes_sbox.txt")).unwrap()
}

/// Produces the circuit that given a message m and a key k,  outputs AES_k(m)
pub fn aes_circuit() -> TCircuit<([bool; 128], [bool; 128]), [bool; 128]> {
    TCircuit::read_from_file_bf(&get_def_circuit("aes_128.txt")).unwrap()
}
