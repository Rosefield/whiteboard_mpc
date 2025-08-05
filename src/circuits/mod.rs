pub type WireId = usize;

pub mod circuit;
pub use circuit::{Circuit, Gate, TCircuit};

pub mod builder;
pub use builder::{new_builder, CircuitBuilder};

pub mod elements;
pub use elements::{CircuitCollection, CircuitElement};

pub mod arith;
pub use arith::CircuitRing;

pub mod executor;

pub mod aes;
pub mod generic_thresh;
pub mod keccak;
pub mod sha256;

static BASE_DIR: &str = "dependencies/circuits";

pub fn get_def_circuit(name: &str) -> String {
    format!("{}/{}", BASE_DIR, name)
}

pub fn bits_le(x: u8) -> [bool; 8] {
    let mut b = [false; 8];
    for i in 0..8 {
        b[i] = (x >> i) & 1 == 1;
    }
    b
}

pub fn bits_be(x: u8) -> [bool; 8] {
    let mut b = [false; 8];
    for i in 0..8 {
        b[7 - i] = (x >> i) & 1 == 1;
    }
    b
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_premade_circuits_well_formed() {
        // Not testing that the circuits output the correct thing
        // Just that they are well-formed.
        let _ = FF2_128::add_circuit().checked();
        let _ = FF2_128::mul_circuit().checked();
        let _ = RR2_128::add_circuit().checked();
        let _ = RR2_128::mul_circuit().checked();
        let _ = aes_circuit().checked();
        let _ = sha256_circuit().checked();
        let _ = hmacsha256_circuit().checked();
        let _ = keccak_circuit().checked();
        let _ = kmac_128().checked();
    }
}
*/
