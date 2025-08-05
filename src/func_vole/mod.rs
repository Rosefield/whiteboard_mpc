use crate::{
    base_func::{CheatOrUnexpectedError, SessionId, UnexpectedError},
    field::{RandElement, Ring},
    linalg::Vector,
    party::PartyId,
};

/// Trait to represent the 2-party VOLE functionality
/// over a ring T.
pub trait AsyncVole {
    /*
    #[type_const]
    const KSI<T: Ring>: usize;
    #[type_const]
    const RHO<T: Ring>: usize;
    */
    /// Start a new instance with `sid`
    async fn init(&self, sid: SessionId, other: PartyId) -> Result<(), UnexpectedError>;

    //const fn allowed_size<T: Ring, const ELL: usize>() -> bool;
    /// calculate $\vec{a} * b$ and give an additive share of the output
    /// to each party.
    async fn input<T: Ring + RandElement, const ELL: usize>(&self, sid: SessionId, other: PartyId, b: T) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
        // I suspect many of these problems would be solved if it was AsyncVole<T> instead of
        // allowing different rings for each call
        // Leak all the implementation details
        where [u8; ELL + 1]: ,
              [u8; ELL + 2]: ;
        /*
        where [(); T::BYTES*8 + 2*80]:,
              [(); (T::BYTES*8).div_ceil(128)]: ,
              [(); ELL + (T::BYTES*8).div_ceil(128)]: ;
        */
        /*
        where [(); Self::KSI::<T>]:,
              [(); Self::RHO::<T>]: ,
              [(); ELL + Self::RHO::<T>]: ;
        */
        /*
        // Attempt arbitrary limits to try to appease the compiler
        where Assert<{ ELL < 1_000_000 }>: IsTrue,
              Assert<{T::BYTES < 1_000_000 }>: IsTrue;
        */

    async fn multiply<T: Ring + RandElement, const ELL: usize>(
        &self,
        sid: SessionId,
        other: PartyId,
        a: &Vector<T, ELL>,
    ) -> Result<Vector<T, ELL>, CheatOrUnexpectedError>
        where [u8; ELL + 1]: ,
              [u8; ELL + 2]: ;
        // Leak all the implementation details
        /*
        where [(); T::BYTES*8 + 2*80]:,
              [(); (T::BYTES*8).div_ceil(128)]: ,
              [(); ELL + (T::BYTES*8).div_ceil(128)]: ;
        */
}

pub mod dkls23_vole;
pub use dkls23_vole::Dkls23VolePlayer;
