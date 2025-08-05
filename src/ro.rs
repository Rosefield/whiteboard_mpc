use crate::field::{RandElement, ToFromBytes};

use rand::{CryptoRng, RngCore};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};

pub struct RO {
    hash: Shake128,
}

struct RoRng {
    reader: <Shake128 as ExtendableOutput>::Reader,
}

impl CryptoRng for RoRng {}

impl RngCore for RoRng {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.reader.read(dst);
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dst);
        Ok(())
    }
}

impl RO {
    pub fn new() -> Self {
        Self {
            hash: Shake128::default(),
        }
    }

    pub fn add_context<T: AsRef<[u8]>>(mut self, context: T) -> Self {
        self.hash.update(context.as_ref());
        self
    }

    pub fn update_context<T: AsRef<[u8]>>(&mut self, context: T){
        self.hash.update(context.as_ref());
    }

    pub fn generate_read<S: ToFromBytes, T: RandElement>(&self, seed: &S) -> T {
        let mut bytes = vec![0u8; S::BYTES];
        seed.to_bytes(&mut bytes);

        self.generate(bytes)
    }

    pub fn generate<S: AsRef<[u8]>, T: RandElement>(&self, seed: S) -> T {
        let h = self.hash.clone();
        let s = h.chain(seed).finalize_xof();

        let mut rng = RoRng { reader: s };

        T::rand(&mut rng)
    }
}
