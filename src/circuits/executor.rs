use crate::circuits::{CircuitCollection, CircuitElement, Gate, TCircuit};

pub fn execute_circuit_inner<I, O, FI: Fn(&I, &mut [bool]), FO: Fn(&[bool]) -> O>(
    input: &I,
    encode_fun: FI,
    circuit: &TCircuit<I, O>,
    parse_fun: FO,
) -> O {
    let num_wires = circuit.inputs.len() + circuit.gates.len();
    let mut wires: Vec<bool> = Vec::with_capacity(num_wires);
    wires.resize(num_wires, false);

    encode_fun(input, wires.as_mut_slice());

    for g in circuit.gates.iter() {
        match g {
            &Gate::And(x, y, z) => {
                wires[z] = wires[x] & wires[y];
            }
            &Gate::Xor(x, y, z) => {
                wires[z] = wires[x] ^ wires[y];
            }
            &Gate::Not(x, y) => {
                wires[y] = !wires[x];
            }
        };
    }

    let outputs: Vec<bool> = circuit.outputs.iter().map(|&o| wires[o]).collect();
    parse_fun(outputs.as_slice())
}

pub fn execute_circuit<I: CircuitElement, O: CircuitElement>(
    input: &I,
    circuit: &TCircuit<I, O>,
) -> O {
    execute_circuit_inner(input, I::to_bits, circuit, O::from_bits)
}

pub fn execute_circuit_multi<I: CircuitCollection, O: CircuitElement>(
    input: &I,
    _num_inputs: usize,
    circuit: &TCircuit<I, O>,
) -> O {
    execute_circuit_inner(input, I::to_bits, circuit, O::from_bits)
}

pub fn execute_circuit_mxm<I: CircuitCollection, O: CircuitCollection>(
    input: &I,
    _num_inputs: usize,
    circuit: &TCircuit<I, O>,
    num_outputs: usize,
) -> O {
    execute_circuit_inner(input, I::to_bits, circuit, |m| O::from_bits(num_outputs, m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuits::{
            aes::{aes_circuit, aes_key_schedule, aes_sbox, aes_with_schedule},
            arith::{linear, sum_circuit, sum_many_circuit, CircuitRing},
            elements::compare_all,
            generic_thresh::setup_circuit,
            keccak::{kmac_128, kmac_128_post, kmac_128_pre},
            sha256::{
                hmacsha256_circuit, hmacsha256_post_circuit, hmacsha256_pre_circuit,
                sha256_circuit, sha_of,
            },
        },
        ff2_128::FF2_128,
        field::{ConstInt, RandElement},
        polynomial::{InterpolationPolynomial, Polynomial},
        rr2_128::RR2_128,
    };

    use rand;

    #[test]
    fn test_ff2_add_cir() {
        let c = FF2_128::add_circuit();
        let x = FF2_128::new(0xFFFF, 0xFFFF);
        let y = FF2_128::new(0xFFFF << 16, 0xFFFF << 16);
        let z = execute_circuit(&(x, y), &c);

        assert!(x + y == z);
    }

    #[test]
    fn test_rr2_add_cir() {
        let c = RR2_128::add_circuit();
        let x = RR2_128::new(0xFFFF, 0xFFFF);
        let y = RR2_128::new(0xFFFF << 16, 0xFFFF << 16);
        let z = execute_circuit(&(x, y), &c);

        assert!(x + y == z);
    }

    #[test]
    fn test_ff2_mul_cir() {
        let c = FF2_128::mul_circuit();
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = FF2_128::rand(&mut rng);
            let y = FF2_128::rand(&mut rng);
            let z = execute_circuit(&(x, y), &c);

            assert!(dbg!(x * y) == dbg!(z));
        }
    }

    #[test]
    fn test_rr2_mul_cir() {
        let c = RR2_128::mul_circuit();
        {
            let x = RR2_128::new(0xFFFF, 0xFFFF);
            let y = RR2_128::new(0xFFFF << 16, 0xFFFF << 16);
            let z = execute_circuit(&(x, y), &c);
            assert!(x * y == z);
        }

        {
            let x = RR2_128::new(0xFFFF, 0xFFFF);
            let y = RR2_128::new(0xFFFF, 0xFFFF);
            let z = execute_circuit(&(x, y), &c);
            assert!(x * y == z);
        }
    }

    #[test]
    fn test_sum_circuit() {
        let c = sum_circuit::<FF2_128>(10);
        let mut rng = rand::thread_rng();
        let xs: Vec<_> = (0..10).map(|_| FF2_128::rand(&mut rng)).collect();

        let z = execute_circuit_multi(&xs, 10, &c);
        assert!(xs.into_iter().sum::<FF2_128>() == z);
    }

    #[test]
    fn test_sum_many_circuit() {
        let c = sum_many_circuit::<FF2_128>(5, 5);
        let mut rng = rand::thread_rng();
        let xs: Vec<_> = (0..25).map(|_| FF2_128::rand(&mut rng)).collect();

        let z = execute_circuit_mxm(&xs, 25, &c, 5);

        let mut res = vec![FF2_128::new(0, 0); 5];
        for c in xs.chunks_exact(5) {
            for (a, b) in res.iter_mut().zip(c) {
                *a += b;
            }
        }

        assert!(res == z);
    }

    #[test]
    fn test_linear() {
        let c = linear::<FF2_128>(10);
        let mut rng = rand::thread_rng();
        let a = FF2_128::rand(&mut rng);
        let xs: Vec<_> = (0..10).map(|_| FF2_128::rand(&mut rng)).collect();

        let encode = |(x, y): &(FF2_128, Vec<FF2_128>), bits: &mut [bool]| {
            x.to_bits(bits);
            y.to_bits(&mut bits[128..]);
        };

        let zs = execute_circuit_inner(&(a, xs.clone()), encode, &c, |b| {
            <Vec<FF2_128> as CircuitCollection>::from_bits(10, b)
        });

        let res: Vec<_> = xs.into_iter().map(|x| x * a).collect();

        assert!(res == zs);
    }

    #[test]
    fn test_compare_all() {
        let c = compare_all::<FF2_128>(10);
        let mut rng = rand::thread_rng();
        let xs: Vec<_> = (0..10).map(|_| FF2_128::rand(&mut rng)).collect();

        let encode = |(x, y): &(Vec<FF2_128>, Vec<FF2_128>), bits: &mut [bool]| {
            x.to_bits(bits);
            y.to_bits(&mut bits[10 * 128..]);
        };

        let b = execute_circuit_inner(&(xs.clone(), xs.clone()), encode, &c, bool::from_bits);

        assert!(b);
    }

    #[test]
    fn test_generic_thresh_setup_circuit() {
        todo!("The comcomp -> GenericThresh migration made the circuits work in terms of bits instead of field elements and this hasn't been fully updated yet.");
        /*
        let p = 2;
        let l = 2;
        let t = 2;
        let points = vec![FF2_128::new(0, 1), FF2_128::new(0, 2)];
        let sum_c = sum_circuit::<FF2_128>(2);
        let c = setup_circuit::<FF2_128, _, _>(p, 128, t, &points, &sum_c);
        let mut rng = rand::thread_rng();
        let xs: Vec<_> = (0..l).map(|_| FF2_128::rand(&mut rng)).collect();
        let mut alpha = FF2_128::zero();

        let mut party_inputs = Vec::new();
        let zero = FF2_128::zero();
        for _ in 0..p {
            let alpha_i = FF2_128::rand(&mut rng);
            let shares_zeros: Vec<_> = (0..2 * l)
                .map(|_| InterpolationPolynomial::secret_share(&mut rng, zero.clone(), t, &points))
                .map(|z| z.vals)
                .flatten()
                .collect();
            alpha += alpha_i;
            party_inputs.push((alpha_i, shares_zeros));
        }

        let expected_out: Vec<_> = xs.iter().map(|x| (x.clone(), x.clone() * &alpha)).collect();

        let encode_fn = |i: &(Vec<FF2_128>, Vec<(FF2_128, Vec<FF2_128>)>), bits: &mut [bool]| {
            i.0.to_bits(bits);
            let mut start = l * 128;
            for j in i.1.iter() {
                j.0.to_bits(&mut bits[start..]);
                j.1.to_bits(&mut bits[start + 128..]);
                start += 128 * (1 + j.1.len());
            }
        };

        let parse_fn = |bits: &[bool]| {
            let mut outs = Vec::new();
            let size = <Vec<(FF2_128, FF2_128)> as CircuitCollection>::total_size(l);
            let mut start = 0;
            for _ in 0..p {
                let out =
                    <Vec<(FF2_128, FF2_128)> as CircuitCollection>::from_bits(l, &bits[start..]);
                outs.push(out);
                start += size;
            }

            outs
        };

        let outputs = execute_circuit_inner(&(xs, party_inputs), encode_fn, &c, parse_fn);

        // recover secret shares from outputs
        let shares: (Vec<Vec<_>>, Vec<Vec<_>>) =
            outputs.into_iter().map(|o| o.into_iter().unzip()).unzip();
        let transpose = |v: Vec<Vec<_>>| {
            let mut out: Vec<Vec<_>> = Vec::new();
            for _ in 0..v[0].len() {
                out.push(Vec::new());
            }

            for ins in v.into_iter() {
                for (i, x) in ins.into_iter().enumerate() {
                    out[i].push(x);
                }
            }

            out
        };
        let (xs_shares, macs_shares) = (transpose(shares.0), transpose(shares.1));
        let xs_out: Vec<_> = xs_shares
            .iter()
            .map(|x| {
                InterpolationPolynomial::new(&points, x)
                    .unwrap()
                    .eval_zero()
            })
            .collect();
        let macs_out: Vec<_> = macs_shares
            .iter()
            .map(|x| {
                InterpolationPolynomial::new(&points, x)
                    .unwrap()
                    .eval_zero()
            })
            .collect();

        let calced_out: Vec<_> = xs_out.into_iter().zip(macs_out.into_iter()).collect();

        assert_eq!(expected_out, calced_out);
        */
    }

    fn bits_be(x: u8) -> [bool; 8] {
        let mut b = [false; 8];
        for i in 0..8 {
            b[7 - i] = (x >> i) & 1 == 1;
        }
        b
    }

    #[test]
    fn test_sha2_circuit() {
        let sha256 = sha256_circuit();

        let mut sha_initial_state = [false; 256];

        let ivs = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];

        ivs.into_iter()
            .zip(sha_initial_state.chunks_exact_mut(32))
            .for_each(|(x, s)| {
                x.to_be_bytes()
                    .into_iter()
                    .zip(s.chunks_exact_mut(8))
                    .for_each(|(b, c)| {
                        c.copy_from_slice(&bits_be(b));
                    })
            });

        let pad = |m: &[u8]| {
            let l = m.len();
            assert!(l < 56);
            let mut paded = [false; 512];
            paded.chunks_exact_mut(8).zip(m.iter()).for_each(|(c, &x)| {
                c.copy_from_slice(&bits_be(x));
            });
            // append a 1 bit
            paded[l * 8] = true;
            // append the count of bits as a 64-bit BE integer
            // but as we restrict the message to be less than a block
            // this requires at most 16 bits
            let nbits = (l * 8) as u16;
            nbits
                .to_be_bytes()
                .into_iter()
                .zip(paded[496..].chunks_exact_mut(8))
                .for_each(|(b, c)| {
                    c.copy_from_slice(&bits_be(b));
                });

            paded
        };

        // For whatever reason, the bristol SHA256 circuit reverses the order
        // of inputs/outputs to be LE compared to the standard BE encoding.
        // e.g. block 10000....00 the circuit expects the input to be 00..001
        sha_initial_state.reverse();
        // 0 len message is entirely pad
        {
            let mut m = pad(b"");
            m.reverse();
            let mut output = execute_circuit(&(m, sha_initial_state), &sha256);
            output.reverse();

            let expected = [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
                0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
                0x78, 0x52, 0xb8, 0x55,
            ];
            let ebits: Vec<_> = expected.into_iter().map(|e| bits_be(e)).flatten().collect();
            assert_eq!(&output, &ebits[..]);
        }

        {
            let mut m = pad(b"abc");
            m.reverse();
            let mut output = execute_circuit(&(m, sha_initial_state), &sha256);
            output.reverse();

            let expected = [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ];
            let ebits: Vec<_> = expected.into_iter().map(|e| bits_be(e)).flatten().collect();
            assert_eq!(&output, &ebits[..]);

            let sha2 = sha_of::<[bool; 24]>();
            let mut m2 = [false; 24];
            m2[..8].copy_from_slice(&bits_be(b'a'));
            m2[8..16].copy_from_slice(&bits_be(b'b'));
            m2[16..].copy_from_slice(&bits_be(b'c'));

            let output = execute_circuit(&m2, &sha2);
            assert_eq!(&output, &ebits[..]);
        }

        {
            let mut m = pad(b"The quick brown fox jumps over the lazy dog");
            m.reverse();
            let mut output = execute_circuit(&(m, sha_initial_state), &sha256);
            output.reverse();

            let expected = [
                0xd7, 0xa8, 0xfb, 0xb3, 0x07, 0xd7, 0x80, 0x94, 0x69, 0xca, 0x9a, 0xbc, 0xb0, 0x08,
                0x2e, 0x4f, 0x8d, 0x56, 0x51, 0xe4, 0x6d, 0x3c, 0xdb, 0x76, 0x2d, 0x02, 0xd0, 0xbf,
                0x37, 0xc9, 0xe5, 0x92,
            ];
            let ebits: Vec<_> = expected.into_iter().map(|e| bits_be(e)).flatten().collect();
            assert_eq!(&output, &ebits[..]);
        }
    }

    #[test]
    fn test_hmac_circuit() {
        let c = hmacsha256_circuit();

        let input = ([false; 256], [false; 256]);
        let output = execute_circuit(&input, &c);

        let expected = [
            0x33, 0xad, 0x0a, 0x1c, 0x60, 0x7e, 0xc0, 0x3b, 0x09, 0xe6, 0xcd, 0x98, 0x93, 0x68,
            0x0c, 0xe2, 0x10, 0xad, 0xf3, 0x00, 0xaa, 0x1f, 0x26, 0x60, 0xe1, 0xb2, 0x2e, 0x10,
            0xf1, 0x70, 0xf9, 0x2a,
        ];
        let ebits: Vec<_> = expected.into_iter().map(|e| bits_be(e)).flatten().collect();

        assert_eq!(&output, &ebits[..]);
    }

    #[test]
    fn test_hmac_split_circuits() {
        let hmac = hmacsha256_circuit();
        let hmac_pre = hmacsha256_pre_circuit();
        let hmac_post = hmacsha256_post_circuit();

        let mut key = [false; 256];
        let mut pt = [false; 256];

        // somewhat varied bit pattern to make sure endianness
        // is handled properly.
        for i in 0..128 {
            key[i] = (i % 3) == 0;
            pt[i] = (i % 2) == 0;
        }

        let exp_out = execute_circuit(&(key, pt), &hmac);

        let pre_out = execute_circuit(&key, &hmac_pre);

        let test_out = execute_circuit(&(pt, pre_out), &hmac_post);

        assert_eq!(exp_out, test_out);
    }

    #[test]
    fn test_aes_sbox_circuit() {
        let sbox = aes_sbox();

        let cases = [
            (0x00, 0x63),
            (0x01, 0x7c),
            (0x3a, 0x80),
            (0xc2, 0x25),
            (0x88, 0xc4),
        ];

        let bits = |x: u8| {
            let mut b = [false; 8];
            for i in 0..8 {
                b[7 - i] = (x >> i) & 1 == 1;
            }
            b
        };

        for (i, o) in cases.into_iter() {
            let is = bits(i);

            let os = execute_circuit(&is, &sbox);

            assert_eq!(bits(o), os);
        }
    }

    #[test]
    fn test_aes_split_circuit() {
        let aes = aes_circuit();
        let aes_schedule = aes_key_schedule();
        let aes_expanded = aes_with_schedule();

        let mut key = [false; 128];
        let mut pt = [false; 128];

        // somewhat varied bit pattern to make sure endianness
        // is handled properly.
        for i in 0..128 {
            key[i] = (i % 3) == 0;
            pt[i] = (i % 2) == 0;
        }

        let exp_out = execute_circuit(&(pt, key), &aes);

        let mut full_schedule = [false; 1408];
        full_schedule[..128].copy_from_slice(&key);
        let schedule = execute_circuit(&key, &aes_schedule);
        full_schedule[128..].copy_from_slice(&schedule);

        let test_out = execute_circuit(&(pt, full_schedule), &aes_expanded);

        assert_eq!(exp_out, test_out);
    }

    #[test]
    fn test_kmac() {
        let c = kmac_128();

        let input = ([false; 256], [false; 256]);
        let output = execute_circuit(&input, &c);

        println!("{:?}", output);
        //assert!(false);
    }

    #[test]
    fn test_kmac_split_circuit() {
        let kmac = kmac_128();
        let kmac_pre = kmac_128_pre();
        let kmac_post = kmac_128_post();

        let key = [false; 256];
        let pt = [false; 256];

        let exp_out = execute_circuit(&(key, pt), &kmac);

        let state = execute_circuit(&key, &kmac_pre);
        let test_out = execute_circuit(&(pt, state), &kmac_post);

        assert_eq!(exp_out, test_out);
    }
}
