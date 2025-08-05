use crate::field::RandElement;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use log::warn;

pub struct Prg;

impl Prg {
    pub fn new() -> Self {
        Self
    }

    pub fn generate<S: AsRef<[u8]>, T: RandElement>(&self, seed: S) -> T {
        let mut cs = [0; 32];
        let seed = seed.as_ref();

        debug_assert!(seed.len() >= 16, "Seed should be at least 16 bytes");
        if seed.len() > 32 {
            warn!("The whole seed provided to Prg::generate was not used (max 32 bytes)");
        }
        let len = usize::min(32, seed.len());
        cs[..len].copy_from_slice(&seed[..len]);

        let mut rng = ChaCha20Rng::from_seed(cs);
        T::rand(&mut rng)
    }
}
