use crate::circuits::{
    bits_be,
    builder::new_builder,
    circuit::{Gate, TCircuit},
    elements::CircuitElement,
    get_def_circuit,
};

/// Produces the circuit that given input m, state s, outputs SHA256(m; s)
/// Where m, s are LITTLE ENDIAN encoded contrary to the standardized big endian
/// encoding.
pub fn sha256_circuit() -> TCircuit<([bool; 512], [bool; 256]), [bool; 256]> {
    TCircuit::read_from_file_bf(&get_def_circuit("sha256_bf.txt")).unwrap()
}

fn sha256_initial_state() -> [bool; 256] {
    let mut state = [false; 256];
    [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ]
    .into_iter()
    .zip(state.chunks_exact_mut(32))
    .for_each(|(x, s)| {
        x.to_be_bytes()
            .into_iter()
            .zip(s.chunks_exact_mut(8))
            .for_each(|(b, c)| {
                c.copy_from_slice(&bits_be(b));
            })
    });

    state
}

pub fn hmacsha256_pre_circuit() -> TCircuit<[bool; 256], [bool; 512]> {
    let sha_c = sha256_circuit();
    // HMAC-SHA256(k, m) = H((k + opad) || H((k + ipad) || m))
    // where the 256-bit key is padded to 512 bits (one block of input) and masked
    // and opad = 0x5c * 64, ipad = 0x36 * 64
    // before the second block is hashed.
    // Thus for |m| <= 512, this requires 4 evaluations of the base SHA256 hash function
    let builder = new_builder().with_consts();

    let (builder, key) = builder.add_input::<[bool; 256]>(None);

    // 0x36 = 00110110
    let ipad: Vec<_> = builder.encode_consts(&bits_be(0x36));
    // 0x5c = 01011100
    let opad: Vec<_> = builder.encode_consts(&bits_be(0x5c));

    let padk_c: TCircuit<([bool; 256], [bool; 8]), [bool; 256]> = {
        let inputs: Vec<_> = (0..264).collect();
        let k = &inputs[..256];
        let p = &inputs[256..];
        let gates = k.iter().map(|i| Gate::Xor(*i, p[i % 8], 264 + i)).collect();

        let outputs = (264..520).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    };

    let mut ko_in = key.clone();
    ko_in.extend(&opad);
    let (builder, mut key_outer) = builder.extend_circuit(&ko_in, &padk_c, None);
    for _ in 0..32 {
        key_outer.extend(&opad);
    }

    let mut ki_in = key;
    ki_in.extend(&ipad);
    let (builder, mut key_inner) = builder.extend_circuit(&ki_in, &padk_c, None);
    for _ in 0..32 {
        key_inner.extend(&ipad);
    }

    let mut sha_initial_state = builder.encode_consts(&sha256_initial_state());

    // the SHA circuit expects the inputs to be in LE order,
    // contrary to the SHA specification that uses BE words/bits
    // so reverse the inputs
    sha_initial_state.reverse();

    let mut hashin_input_1 = key_inner;
    hashin_input_1.reverse();
    hashin_input_1.extend(&sha_initial_state);

    let (builder, hash_in_1) = builder.extend_circuit(&hashin_input_1, &sha_c, None);

    let mut hashout_input_1 = key_outer;
    hashout_input_1.reverse();
    hashout_input_1.extend(sha_initial_state);
    let (builder, mut hash_out_1) = builder.extend_circuit(&hashout_input_1, &sha_c, None);
    let mut hashes = hash_in_1;
    hashes.append(&mut hash_out_1);

    builder.refine_input().refine_output(&hashes).to_circuit()
}

fn sha_pad<I: CircuitElement>() -> Vec<bool> {
    let l = I::BIT_SIZE;

    // pad consists of 1 || 0^* || enc(l) with 0s to make the length divisible by 512
    let pad_size = 512 - ((l + 65) % 512);

    let mut pad = vec![false; pad_size + 65];
    pad[0] = true;

    let pl = pad.len();
    pad[pl - 64..]
        .chunks_exact_mut(8)
        .zip(l.to_be_bytes().into_iter())
        .for_each(|(p, b)| {
            p.copy_from_slice(&bits_be(b));
        });

    return pad;
}

pub fn hmacsha256_post_circuit() -> TCircuit<([bool; 256], [bool; 512]), [bool; 256]> {
    let sha_c = sha256_circuit();
    let builder = new_builder().with_consts();

    let (builder, msg) = builder.add_input::<[bool; 256]>(None);
    let (builder, keys) = builder.add_input::<[bool; 512]>(None);
    let hash_in_1 = &keys[..256];
    let hash_out_1 = &keys[256..];

    // hash input is padded with the bit 1, 191 bits 0, and then the count of bits hashed
    // as 8 BE bytes. Here the count of bits is always 768 (512b padded key + 256b message)
    let pad = builder.encode_consts(&sha_pad::<[bool; 768]>());

    // finish H(k + ipad || m)
    let mut hashin_input_2 = msg;
    hashin_input_2.extend(&pad);
    hashin_input_2.reverse();
    hashin_input_2.extend(hash_in_1);
    let (builder, mut hash_in_2) = builder.extend_circuit(&hashin_input_2, &sha_c, None);
    hash_in_2.reverse();

    // finish H(k + opad || H(...))
    let mut hashout_input_2 = hash_in_2;
    hashout_input_2.extend(pad);
    hashout_input_2.reverse();
    hashout_input_2.extend(hash_out_1);
    let (builder, mut hmac) = builder.extend_circuit(&hashout_input_2, &sha_c, None);
    hmac.reverse();

    builder.refine_input().refine_output(&hmac).to_circuit()
}

pub fn sha_of<I: CircuitElement>() -> TCircuit<I, [bool; 256]> {
    let sha_c = sha256_circuit();

    let builder = new_builder().with_consts();

    let (mut builder, input) = builder.add_input::<I>(None);

    let mut blocks = input;
    blocks.append(&mut builder.encode_consts(&sha_pad::<I>()));

    let mut state = builder.encode_consts(&sha256_initial_state());

    state.reverse();

    for b in blocks.chunks_exact(512) {
        let mut b_in = b.to_vec();
        b_in.reverse();
        b_in.append(&mut state);

        (builder, state) = builder.extend_circuit(&b_in, &sha_c, None);
    }

    state.reverse();

    builder.refine_input().refine_output(&state).to_circuit()
}

/// Produces the circuit that given key k, and message m, outputs HMAC-SHA256(key, m)
pub fn hmacsha256_circuit() -> TCircuit<([bool; 256], [bool; 256]), [bool; 256]> {
    let sha_c = sha_of::<[bool; 768]>();
    // HMAC-SHA256(k, m) = H((k + opad) || H((k + ipad) || m))
    // where the 256-bit key is padded to 512 bits (one block of input) and masked
    // and opad = 0x5c * 64, ipad = 0x36 * 64
    // before the second block is hashed.
    // Thus for |m| <= 512, this requires 4 evaluations of the base SHA256 compression function
    let builder = new_builder().with_consts();

    let (builder, key) = builder.add_input::<[bool; 256]>(None);
    let (builder, msg) = builder.add_input::<[bool; 256]>(None);

    // constants are BE
    // 0x36 = 00110110
    let ipad: Vec<_> = builder.encode_consts(&bits_be(0x36));
    // 0x5c = 01011100
    let opad: Vec<_> = builder.encode_consts(&bits_be(0x5c));

    let padk_c: TCircuit<([bool; 256], [bool; 8]), [bool; 256]> = {
        let inputs: Vec<_> = (0..264).collect();
        let k = &inputs[..256];
        let p = &inputs[256..];
        let gates = k.iter().map(|i| Gate::Xor(*i, p[i % 8], 264 + i)).collect();

        let outputs = (264..520).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    };

    let mut ko_in = key.clone();
    ko_in.extend(&opad);
    let (builder, mut key_outer) = builder.extend_circuit(&ko_in, &padk_c, None);
    for _ in 0..32 {
        key_outer.extend(&opad);
    }

    let mut ki_in = key;
    ki_in.extend(&ipad);
    let (builder, mut key_inner) = builder.extend_circuit(&ki_in, &padk_c, None);
    for _ in 0..32 {
        key_inner.extend(&ipad);
    }

    let mut hashin_input = key_inner;
    hashin_input.extend(&msg);
    let (builder, hash_in) = builder.extend_circuit(&hashin_input, &sha_c, None);

    let mut hashout_input = key_outer;
    hashout_input.extend(&hash_in);
    let (builder, hmac) = builder.extend_circuit(&hashout_input, &sha_c, None);

    builder.refine_input().refine_output(&hmac).to_circuit()
}
