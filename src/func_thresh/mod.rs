use crate::{
    base_func::UnexpectedError,
    circuits::{CircuitElement, TCircuit},
    party::PartyId,
};

pub type InputId = usize;

/// The trait to represent the F_thresh functionality over base element type T.
/// Does not allow for concurrent operations.
pub trait AsyncThresh {
    /// Restore the saved keys and inputs from a previous run, replacing the `init` call.
    async fn resume_from_state_file(
        &mut self,
        state_file: &str,
        run_init: bool,
    ) -> Result<(), UnexpectedError>;

    /// Save the secret key and input shares to `state_file`
    fn write_state_to_file(&self, state_file: &str) -> Result<(), UnexpectedError>;

    /// Runs the initial setup for the functionality to generate keys
    async fn init(&mut self) -> Result<(), UnexpectedError>;

    /// Jointly run a circuit to generate a threshold-shared state
    /// Each party provides k inputs such that |I| = nk
    /// and generates |O| threshold auth bits using the circuit output.
    async fn setup<I, O: CircuitElement>(
        &mut self,
        input: &[bool],
        out_ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<(), UnexpectedError>;

    /// Collectively sample |ids| random inputs
    async fn sample(&mut self, ids: &[InputId]) -> Result<(), UnexpectedError>;

    /// Evaluate the given circuit with the specified list of parties
    async fn eval<I, O: CircuitElement>(
        &mut self,
        parties: &[PartyId],
        ids: &[InputId],
        circuit: &TCircuit<I, O>,
    ) -> Result<O, UnexpectedError>;
}

pub mod rst_thresh;
pub use rst_thresh::RstThreshPlayer;

pub mod generic_thresh;
pub use generic_thresh::GenericThreshPlayer;
