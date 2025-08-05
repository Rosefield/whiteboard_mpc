use crate::{
    base_func::{FuncId, SessionId},
    func_net::AsyncNet,
    party::PartyId,
};

use std::sync::Arc;

struct NetShimGeneric<T: AsyncNet>(Arc<T>);

trait _NetSync: Send {
    fn send_to(&self, dst: i32, func: u16, data: &[u8]);
    fn recv_from(&self, dst: i32, func: u16, data: &mut [u8]) -> u32;
}

pub fn sync<F: std::future::Future>(f: F) -> F::Output {
    let handle = tokio::runtime::Handle::try_current();

    if let Ok(h) = handle {
        h.block_on(f)
    } else {
        // making a new runtime each time is _probably_ bad for perf
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        println!("new runtime");
        let o = rt.block_on(f);
        o
    }
}

impl<T: AsyncNet> _NetSync for NetShimGeneric<T> {
    fn send_to(&self, dst: i32, func: u16, data: &[u8]) {
        let dst = dst as PartyId;
        let func: FuncId = func.into();
        sync(self.0.send_to_local(dst, func, SessionId::new(func), data)).unwrap();
    }

    fn recv_from(&self, dst: i32, func: u16, data: &mut [u8]) -> u32 {
        let dst = dst as PartyId;
        let func: FuncId = func.into();
        let (_, n) = sync(
            self.0
                .recv_from_local(dst, func, SessionId::new(func), data),
        )
        .unwrap();
        n as u32
    }
}

pub struct NetShim {
    net: Box<dyn _NetSync>,
}

impl NetShim {
    pub fn from_net<T: AsyncNet>(n: Arc<T>) -> Self {
        Self {
            net: Box::new(NetShimGeneric(n)),
        }
    }
    pub fn send_to(&self, dst: i32, func: u16, data: &[u8]) {
        self.net.send_to(dst, func, data);
    }
    pub fn recv_from(&self, dst: i32, func: u16, data: &mut [u8]) -> u32 {
        self.net.recv_from(dst, func, data)
    }
}

#[cxx::bridge]
pub mod ffi {
    struct PartyInfo {
        id: u16,
        ip: String,
        port: u16,
    }

    struct Gate {
        in1: i32,
        in2: i32,
        out: i32,
        // 0 | 1 | 2 for AND/XOR/NOT
        kind: i32,
    }

    struct Abit<'mpc> {
        bit: bool,
        keys: &'mpc [[u8; 16]],
        macs: &'mpc [[u8; 16]],
    }

    struct MpcOut {
        outs: Vec<bool>,
        auth_outs: Vec<u8>,
        bytes_sent: u64,
    }

    extern "Rust" {
        fn spawn(f: UniquePtr<Function>);

        type NetShim;
        fn send_to(self: &NetShim, dst: i32, func: u16, data: &[u8]);
        fn recv_from(self: &NetShim, dst: i32, func: u16, data: &mut [u8]) -> u32;
    }

    // emp-tool FFI
    unsafe extern "C++" {
        include!("whiteboard_mpc/dependencies/mpc_ffi.h");

        type Function;
        fn call(f: UniquePtr<Function>);

        type Network;
        fn make_network(
            my_id: u16,
            parties: &[PartyInfo],
            port_offset: u16,
        ) -> Result<SharedPtr<Network>>;

        type IknpOte;
        fn make_ot_player(
            my_id: u16,
            other_id: u16,
            net: SharedPtr<Network>,
            is_sender: bool,
            delta: [u8; 16],
        ) -> Result<SharedPtr<IknpOte>>;

        fn ote_extend_send_rand(self: &IknpOte, out_corr: &mut [[u8; 16]]) -> Result<()>;

        fn ote_extend_recv_rand(
            self: &IknpOte,
            selection: &[bool],
            out_blocks: &mut [[u8; 16]],
        ) -> Result<()>;

        fn net_stat(self: &IknpOte) -> u64;

        type EmpAbit;
        fn make_abit_player(
            my_id: u16,
            parties: &[u16],
            net: SharedPtr<Network>,
            delta: &[bool],
        ) -> Result<SharedPtr<EmpAbit>>;

        // Returns the number of bytes sent
        fn create_abits(
            self: &EmpAbit,
            bits: &[bool],
            out_macs: &mut [&mut [[u8; 16]]],
            out_keys: &mut [&mut [[u8; 16]]],
        ) -> Result<u64>;

        type BristolFormat;

        fn make_bristol_circuit(
            num_inputs: i32,
            num_outputs: i32,
            gates: Vec<Gate>,
        ) -> UniquePtr<BristolFormat>;

        fn run_mpc<'mpc>(
            my_id: u16,
            parties: &[u16],
            net: SharedPtr<Network>,
            delta: Vec<bool>,
            input_assignment: Vec<i32>,
            input: Vec<bool>,
            auth_inputs: Vec<Abit<'mpc>>,
            output_assignment: Vec<i32>,
            circuit: UniquePtr<BristolFormat>,
        ) -> Result<MpcOut>;
    }

    // lib-ote FFI
    unsafe extern "C++" {}
}

unsafe impl Send for ffi::BristolFormat {}
unsafe impl Send for ffi::Function {}

// It definitely isn't, but we can pretend
unsafe impl Send for ffi::EmpAbit {}
unsafe impl Sync for ffi::EmpAbit {}

unsafe impl Send for ffi::IknpOte {}
unsafe impl Sync for ffi::IknpOte {}

unsafe impl Send for ffi::Network {}
unsafe impl Sync for ffi::Network {}

impl std::fmt::Debug for ffi::EmpAbit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmpAbit {{ }}")
    }
}

impl std::fmt::Debug for ffi::IknpOte {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IknpOte {{ }}")
    }
}

impl std::fmt::Debug for ffi::Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmpNetwork {{ }}")
    }
}

pub fn spawn(f: cxx::UniquePtr<ffi::Function>) {
    let _ = tokio::task::spawn_blocking(move || {
        ffi::call(f);
    });
}
