use crate::circuits::{
    arith::CircuitRing,
    bits_le,
    builder::{new_builder, CircuitBuilder},
    circuit::TCircuit,
    elements::CircuitElement,
    get_def_circuit, WireId,
};

pub const B: usize = 1600;

/// Produces the circuit that given state s, outputs state keccak_f(s)
pub fn keccak_circuit() -> TCircuit<[bool; B], [bool; B]> {
    TCircuit::read_from_file_bf(&get_def_circuit("keccak.txt")).unwrap()
}

fn pad101(r: usize, l: usize) -> Vec<bool> {
    let j = r - ((l + 2) % r);
    let mut pad = vec![false; j + 2];
    pad[0] = true;
    pad[j + 1] = true;
    pad
}

fn left_encode(x: u8) -> [bool; 16] {
    let mut bits = [false; 16];
    bits[..8].copy_from_slice(&bits_le(1));
    bits[8..].copy_from_slice(&bits_le(x));
    bits
}

fn right_encode(x: u16) -> Vec<bool> {
    let n = x / 256 + 1;
    let bytes = x.to_le_bytes();

    let mut bits: Vec<bool> = Vec::new();
    for i in 0..n {
        bits.extend(&bits_le(bytes[i as usize]));
    }
    bits.extend(&bits_le(n as u8));
    bits
}

fn byte_pad<I, O>(x: &[WireId], l: u8, builder: &CircuitBuilder<I, O>) -> Vec<WireId> {
    let consts = builder.get_const_wire_ids().unwrap();
    let mut le = builder.encode_consts(&left_encode(l));
    le.extend(x);

    le.resize((l as usize) * 8, consts[0]);
    le
}

fn encode_string(s: &[u8]) -> Vec<bool> {
    let mut bits = left_encode(s.len() as u8).to_vec();

    for &c in s.iter() {
        bits.extend(&bits_le(c));
    }

    bits
}

pub fn keccak_of<I: CircuitElement, const C: usize, const O: usize>(
    state: Option<Vec<bool>>,
) -> TCircuit<I, [bool; O]>
where
    [(); B - C]:,
{
    assert_eq!(C % 8, 0);
    assert!(C <= 1024);

    let keccak_circuit = keccak_circuit();
    let r = B - C;

    let builder = new_builder().with_consts();
    let (mut builder, mut input) = builder.add_input::<I>(None);

    input.append(&mut builder.encode_consts(&pad101(r, input.len())));

    //Step 5
    let mut s = builder.encode_consts(&state.unwrap_or(vec![false; B]));

    // apply the permutation for each block of input
    let xor_circuit = <[bool; B - C] as CircuitRing>::add_circuit();

    for bi in input.chunks_exact(r) {
        let mut xor_ids = s[..r].to_vec();
        xor_ids.extend(bi);
        let (builder2, mut state_out) = builder.extend_circuit(&xor_ids[..], &xor_circuit, None);
        state_out.extend(&s[r..]);
        let (builder2, keccak_out) = builder2.extend_circuit(&state_out[..], &keccak_circuit, None);

        s = keccak_out;
        builder = builder2;
    }

    // for now, don't have extendable output
    assert!(O <= r);

    return builder.refine_input().refine_output(&s[..O]).to_circuit();
}

pub fn kmac_128_initial_state() -> Vec<bool> {
    /*
    use crate::circuits::executor::execute_circuit;
    //calculate byte_pad(encode("KMAC") || encode(""), 168)
    let mut ns = Vec::new();
    ns.extend(left_encode(168));
    ns.append(&mut encode_string(b"KMAC"));
    // the S customization string, default to empty
    ns.append(&mut encode_string(b""));
    ns.resize(168*8, false);

    let mut n = [false; 1600];
    n[..1344].copy_from_slice(&ns[..]);

    let c = keccak_circuit();

    let o = execute_circuit(&n, &c);

    return o.to_vec();
    */

    let s: Vec<_> = [
        0x98, 0x44, 0x06, 0xeb, 0x43, 0x8c, 0x89, 0x6b, 0x4b, 0xc3, 0x48, 0x80, 0x12, 0x34, 0x72,
        0x99, 0x3a, 0xee, 0xe4, 0x51, 0xe6, 0x75, 0x5b, 0x3f, 0xc4, 0x73, 0x31, 0xe5, 0xfe, 0x01,
        0x23, 0x75, 0xf6, 0xfa, 0xdd, 0x1b, 0xa6, 0x82, 0x73, 0xb0, 0x9c, 0xf6, 0xe4, 0x9f, 0xdb,
        0x13, 0x31, 0xc3, 0x6a, 0xf8, 0xcd, 0x6e, 0x92, 0x74, 0x82, 0x59, 0x5c, 0x98, 0x62, 0xd1,
        0xfe, 0xab, 0x49, 0xd1, 0xa2, 0x90, 0x7d, 0x67, 0xcf, 0xaa, 0xe5, 0xa2, 0xd6, 0xc1, 0xd7,
        0x5c, 0x69, 0x5c, 0xab, 0x1f, 0x26, 0x74, 0x65, 0xd0, 0x8c, 0xda, 0x04, 0xb3, 0xd7, 0x68,
        0xf7, 0xb7, 0x0b, 0x4c, 0xac, 0x26, 0xcc, 0x1e, 0x4b, 0xcf, 0x77, 0x23, 0x67, 0x03, 0x2d,
        0xcb, 0xb4, 0x72, 0xcf, 0x21, 0x01, 0x84, 0xa2, 0x5d, 0xb6, 0xfe, 0x8c, 0xad, 0xfe, 0x9e,
        0x57, 0xbb, 0x2e, 0x59, 0x4e, 0xe7, 0x9f, 0xca, 0x69, 0x3e, 0xdc, 0xe0, 0xb5, 0x29, 0xdf,
        0x85, 0xd3, 0xab, 0x0d, 0xf6, 0xef, 0xd3, 0xe4, 0xf2, 0x65, 0xa5, 0x59, 0xcd, 0x66, 0xbd,
        0xfd, 0xde, 0xa2, 0xac, 0xbb, 0xac, 0xe3, 0x84, 0x47, 0x9d, 0x81, 0xfd, 0xaf, 0xdd, 0x44,
        0xd0, 0x2a, 0x03, 0x11, 0x56, 0x36, 0x7c, 0x52, 0x47, 0x97, 0x81, 0xc3, 0xd3, 0xf9, 0x54,
        0xfd, 0xcc, 0xf9, 0xda, 0x40, 0x3b, 0x3a, 0xe5, 0x2d, 0xb2, 0x27, 0x57, 0xed, 0x65, 0x34,
        0x22, 0x97, 0x88, 0xd2, 0x25,
    ]
    .into_iter()
    .map(|b| bits_le(b))
    .flatten()
    .collect();

    s
}

pub fn kmac_128_pre() -> TCircuit<[bool; 256], [bool; B]> {
    let keccak_circuit = keccak_circuit();
    // Input to keccak_f
    // N = "KMAC"
    // S = ""
    let r = B - 256;

    let builder = new_builder().with_consts();

    //wire ids for k
    let (mut builder, key_wires) = builder.add_input::<[bool; 256]>(None);

    // kmac_half = bytepad(encode_string(N) || encode_string(S), 168) || bytepad(encode_string(K), 168)
    // the state after evaluating on the first bytepad(N || S) block
    let mut s = builder.encode_consts(&kmac_128_initial_state());
    let kmac_half = byte_pad(&key_wires, 168, &builder);

    // apply the permutation for each block of input
    let xor_circuit = <[bool; B - 256] as CircuitRing>::add_circuit();

    for bi in kmac_half.chunks_exact(r) {
        let mut xor_ids = s[..r].to_vec();
        xor_ids.extend(bi);
        let (builder2, mut state_out) = builder.extend_circuit(&xor_ids[..], &xor_circuit, None);
        state_out.extend(&s[r..]);
        let (builder2, keccak_out) = builder2.extend_circuit(&state_out[..], &keccak_circuit, None);

        s = keccak_out;
        builder = builder2;
    }

    return builder.refine_input().refine_output(&s).to_circuit();
}

pub fn kmac_128_post() -> TCircuit<([bool; 256], [bool; B]), [bool; 256]> {
    let keccak_circuit = keccak_circuit();
    let r = B - 256;

    let builder = new_builder().with_consts();
    let consts = builder.get_const_wire_ids().unwrap();
    let (builder, mut input) = builder.add_input::<[bool; 256]>(None);
    let (mut builder, state) = builder.add_input::<[bool; B]>(None);

    // KECCAK[256](state; X || right_encode(L) || 00, L)
    input.append(&mut builder.encode_consts(&right_encode(256)));
    input.extend(&[consts[0], consts[0]]);

    let pad = pad101(r, input.len());

    input.append(&mut builder.encode_consts(&pad));

    //Step 5
    let mut s = state;

    // apply the permutation for each block of input
    let xor_circuit = <[bool; B - 256] as CircuitRing>::add_circuit();

    for bi in input.chunks_exact(r) {
        let mut xor_ids = s[..r].to_vec();
        xor_ids.extend(bi);
        let (builder2, mut state_out) = builder.extend_circuit(&xor_ids[..], &xor_circuit, None);
        state_out.extend(&s[r..]);
        let (builder2, keccak_out) = builder2.extend_circuit(&state_out[..], &keccak_circuit, None);

        s = keccak_out;
        builder = builder2;
    }

    return builder.refine_input().refine_output(&s[..256]).to_circuit();
}

pub fn kmac_128() -> TCircuit<([bool; 256], [bool; 256]), [bool; 256]> {
    // Input to keccak_f
    // L = 256
    // N = "KMAC"
    // S = ""
    const R: usize = B - 256;

    let builder = new_builder().with_consts();
    let consts = builder.get_const_wire_ids().unwrap();

    //wire ids for k
    let (builder, key_wires) = builder.add_input::<[bool; 256]>(None);

    //wire id for x
    let (builder, plaintext_wires) = builder.add_input::<[bool; 256]>(None);

    // newX = bytepad(encode_string(K), 168) || X || right_encode(L)
    let mut new_x = byte_pad(&key_wires, 168, &builder);
    new_x.extend(&plaintext_wires);
    new_x.extend(&builder.encode_consts(&right_encode(256)));

    // KECCAK[256](bytepad(encode_string(N) || encode_string(S), 168) || newX || 00, L)
    // the state after processing the bytepad(N || S) string
    let state = kmac_128_initial_state();
    new_x.extend(&[consts[0], consts[0]]);

    let keccak = keccak_of::<[bool; R + 256 + 26], 256, 256>(Some(state));

    let (builder, output) = builder.extend_circuit(&new_x, &keccak, None);

    return builder.refine_input().refine_output(&output).to_circuit();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_kmac_state() {
        let s = kmac_128_initial_state();

        let bytes: Vec<_> = s
            .chunks_exact(8)
            .map(|c| {
                let mut x = 0u8;
                for i in 0..8 {
                    x ^= (if c[i] { 1 } else { 0 }) << i;
                }
                x
            })
            .collect();

        println!("{:02x?}", bytes);
    }
}
