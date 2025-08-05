use crate::{
    field::ConstInt,
    party::{PartyId, PartyInfo},
};

use std::{
    collections::HashMap,
    convert::TryInto,
    fs::File,
    io::{BufRead, BufReader, Write},
    net::ToSocketAddrs,
    str::FromStr,
};

pub fn parse_party_file(file_name: &str) -> (u16, Vec<PartyInfo>) {
    let file = File::open(file_name).unwrap();
    let b = BufReader::new(file);

    let info: Vec<_> = b
        .lines()
        .map(|l| {
            let line = l.unwrap();
            let (id, s) = line.split_once(',').unwrap();
            let (addr, port) = s.split_once(',').unwrap();
            let port = u16::from_str(port).unwrap();
            // EMP networking assumes ipv4
            let addr = (addr, port)
                .to_socket_addrs()
                .unwrap()
                .filter(|a| a.is_ipv4())
                .next()
                .unwrap();
            PartyInfo {
                id: PartyId::from_str(id).unwrap(),
                ip: addr.ip(),
                port: port,
            }
        })
        .collect();

    (info.len() as u16, info)
}

pub fn write_party_file(file_name: &str, party_info: &[PartyInfo]) {
    let mut file = File::create(file_name).unwrap();

    for p in party_info.iter() {
        writeln!(&mut file, "{},{},{}", p.id, p.ip, p.port).unwrap();
    }
}

pub fn write_inputs_file(
    file_name: &str,
    input_assignment: &HashMap<usize, PartyId>,
    my_input: &[bool],
) {
    let mut file = File::create(file_name).unwrap();

    let num_ins = input_assignment.len();
    // number input wires
    writeln!(&mut file, "{num_ins}").unwrap();

    // second line is the assignment of input bits to parties
    let a: Vec<_> = (0..num_ins)
        .map(|i| input_assignment[&i].to_string())
        .collect();
    writeln!(&mut file, "{}", a.join(" ")).unwrap();

    // third line is my input as 0/1s
    let bs: Vec<_> = my_input
        .iter()
        .map(|i| if *i { "1" } else { "0" })
        .collect();
    writeln!(&mut file, "{}", bs.join("")).unwrap();
}

pub fn write_outputs_file(file_name: &str, output_assignment: &HashMap<usize, PartyId>) {
    let mut file = File::create(file_name).unwrap();

    let num_outs = output_assignment.len();
    // number input wires
    writeln!(&mut file, "{num_outs}").unwrap();

    // second line is the assignment of output bits to parties
    let a: Vec<_> = (0..num_outs)
        .map(|i| output_assignment[&i].to_string())
        .collect();
    writeln!(&mut file, "{}", a.join(" ")).unwrap();
}

pub fn serialize_vals_hom<T: ConstInt>(vals: &Vec<T>, sid: u32, size_hint: usize) -> Vec<u8> {
    let mut serialized = vec![0u8; 9 + size_hint * vals.len()];
    serialized[..4].copy_from_slice(&sid.to_le_bytes());
    serialized[4..8].copy_from_slice(&(vals.len() as u32).to_le_bytes());
    // 0 for homogeneous, 1 for heterogeneous
    serialized[8] = 0;
    for i in 0..vals.len() {
        let start = 9 + i * size_hint;
        vals[i].to_bytes(&mut serialized[start..start + size_hint]);
    }

    serialized
}

pub fn serialize_vals_het<T: ConstInt>(vals: &Vec<T>, sid: u32) -> Vec<u8> {
    // we use 2 bytes to give the size for each element v in vals, for now we don't have anything larger than 2^16 bytes
    let num_bytes: usize = vals.iter().map(|v| v.num_bytes() + 2).sum();
    let mut serialized = vec![0u8; 9 + num_bytes];
    serialized[..4].copy_from_slice(&sid.to_le_bytes());
    serialized[4..8].copy_from_slice(&(vals.len() as u32).to_le_bytes());
    // 0 for homogeneous, 1 for heterogeneous
    // serialized as [(len, bytes)]*
    serialized[8] = 1;
    let mut start = 9;
    for v in vals.iter() {
        let size = v.to_bytes(&mut serialized[start + 2..]);
        serialized[start..start + 2].copy_from_slice(&(size as u16).to_le_bytes());

        start += size + 2;
    }

    serialized
}

pub fn deserialize_pairs<T: ConstInt>(bytes: &[u8]) -> (u32, Vec<(T, T)>) {
    let sid = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let vec_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert!(vec_size % 2 == 0);

    if vec_size == 0 {
        return (sid, Vec::new());
    }

    let vals = match bytes[8] {
        0 => {
            let el_size = (bytes.len() - 9) / (vec_size);
            let opts = if el_size == 0 {
                vec![(T::zero(), T::zero()); vec_size / 2]
            } else {
                bytes[9..]
                    .chunks_exact(2 * el_size)
                    .map(|c| {
                        let m0 = T::from_bytes(&c[..el_size]);
                        let m1 = T::from_bytes(&c[el_size..]);
                        (m0, m1)
                    })
                    .collect()
            };
            opts
        }
        1 => {
            let mut vals = Vec::with_capacity(vec_size / 2);

            let mut i = 9;
            while i < bytes.len() {
                let size = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
                let m0 = T::from_bytes(&bytes[i + 2..i + 2 + size]);
                i += 2 + size;
                let size = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
                let m1 = T::from_bytes(&bytes[i + 2..i + 2 + size]);
                vals.push((m0, m1));
                i += 2 + size;
            }
            assert_eq!(i, bytes.len());

            vals
        }
        _ => panic!("Unexpected serialization type"),
    };

    (sid, vals)
}

pub fn deserialize_vals<T: ConstInt>(bytes: &[u8]) -> (u32, Vec<T>) {
    let sid = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let vec_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;

    if vec_size == 0 {
        return (sid, Vec::new());
    }

    let vals = match bytes[8] {
        0 => {
            let el_size = bytes.len() - 9 / vec_size;
            let opts = if el_size == 0 {
                vec![T::zero(); vec_size]
            } else {
                bytes[9..]
                    .chunks_exact(el_size)
                    .map(|c| {
                        let m0 = T::from_bytes(&c[..el_size]);
                        m0
                    })
                    .collect()
            };
            opts
        }
        1 => {
            let mut vals = Vec::with_capacity(vec_size);

            let mut i = 9;
            while i < bytes.len() {
                let size = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
                let el = T::from_bytes(&bytes[i + 2..i + 2 + size]);
                vals.push(el);
                i += 2 + size;
            }

            vals
        }
        _ => panic!("Unexpected serialization type"),
    };

    (sid, vals)
}

pub fn remove_if<T1, T2, F>(vals: &mut Vec<T1>, checks: &mut Vec<T2>, pred: F)
where
    F: Fn(&T2, &T1) -> bool,
{
    assert_eq!(checks.len(), vals.len());
    let mut i = 0;
    while i < checks.len() {
        // if the base is 0 then we found a factor of N and we want to remove that sample
        if pred(&checks[i], &vals[i]) {
            checks.swap_remove(i);
            vals.swap_remove(i);
        } else {
            i += 1;
        }
    }
}
