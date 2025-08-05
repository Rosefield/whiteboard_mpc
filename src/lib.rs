// For circuit functions to generically operate on the number of input/output wires
#![feature(generic_const_exprs)]
// to try to do generic arithmetic on array sizes
#![feature(min_generic_const_args)]
#![feature(generic_const_items)]
// to get zeroed buffers for FFI
#![feature(new_zeroed_alloc)]
// for macroed arithmetic implementations
#![feature(macro_metavar_expr_concat)]
// While prototyping
#![allow(async_fn_in_trait)]
#![allow(non_snake_case)]
//#![allow(unused_variables)]
//#![allow(unreachable_code)]
//#![allow(dead_code)]
//
// For thiserror to support backtraces
#![feature(error_generic_member_access)]
#![feature(test)]
extern crate test;

//extern crate crossbeam;
//extern crate num_bigint;
extern crate rand;
//extern crate rayon;
extern crate serde;
extern crate serde_json;
extern crate sha2;
extern crate tokio;

mod ffi;

pub mod prg;
pub mod punctured_prf;
pub mod ro;

pub mod auth_bits;
pub mod circuits;
pub mod utils;

pub mod ecgroup;
pub mod ff2_128;
pub mod field;
pub mod field_macros;
pub mod linalg;
pub mod p256;
pub mod polynomial;
pub mod rr2_128;
pub mod small_fields;

pub mod base_func;
pub mod func_abit;
pub mod func_com;
pub mod func_cote;
pub mod func_ecdsa;
pub mod func_eot;
pub mod func_mpc;
pub mod func_mult;
pub mod func_net;
pub mod func_rand;
pub mod func_thresh;
pub mod func_thresh_abit;
pub mod func_vole;
pub mod func_zero;
pub mod func_zk;
pub mod party;

pub mod common_protos;
