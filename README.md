# Whiteboard MPC

This library provides an implementation of multiple common (multiparty) protocols in MPC initially designed to implement the dishonest majority threshold MPC scheme of Rosefield, shelat, and Tyner ([RsT24](https://eprint.iacr.org/2024/316)).
The goal of this project is to assist cryptographers in sketching out and developing novel cryptographic protocols by providing many of the tools (e.g. Functionalities) needed to do so.

## Installation 

Compilation requires a Nightly rust toolchain (version >=1.75.0), as well as the EMP-tool and EMP-OT libraries, that themselves depend on OpenSSL.

EMP-tool can be installed with the following commands, see [https://github.com/emp-toolkit/emp-readme](https://github.com/emp-toolkit/emp-readme) for more information.
```
wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py
python[3] install.py --deps --tool --ot
```
The [EMP-agmpc](https://github.com/emp-toolkip/emp-agmpc) library has been modified to interoperate with this library and the modified code can be found in [dependencies/emp_agmpc](dependencies/emp_agmpc)

The library can then be built using cargo
```
cargo build [--release]
```

## Architecture

The library is designed to mimic the Universal Composability (UC) model where we have a number of functionalities represented by traits, and then protocols that realize those functionalities by implementing the trait.
There are a few base concepts:
- `PartyId` that identifies a party in a protocol with a corresponding `PartyInfo` for communication information.
- `FuncId` that identifies the type of functionality, and is associated with the role that a party takes in a (sub)protocol. 
- `SessionId` that identifies what instance of a protocol is being communicated with.
- The `BaseFunc` trait that identifies the `FuncId` and `PartyId` of a functionality instance, and also a static list of `FuncId` for any dependencies.

Because a substantial amount of an MPC protocol relies on communication most functionalities will either directly require or have a transient dependency on the network.
The network, identified by `FuncId::Fnet`, is represented by the `AsyncNet` trait that allows users to (concurrently) send messages to `(PartyId, FuncId)` pairs. 
As such the default implementation (`AsyncNetMgr`) assumes a unique channel for sending to and receiving from `(PartyId, FuncId)`.
This simplified model allows separate functionalities to utilize the network without contention, but does not handle the case where a functionalities with multiple concurrent `SessionId`s wishes to multiplex messages on the same channel.
Currently this is left to the implementer of the functionality if required, but may be implemented in a more standard way in the future.

For example, consider a simplified `AsyncCom` trait that represents the Fcom functionality.
```
pub trait AsyncCom {
    async fn commit_to(self, sid: SessionId, party: PartyId, data: &[u8]) -> Result<(), UnexpectedError>;
    async fn expect_from(self, sid: SessionId, party: PartyId) -> Result<(), _>;
    async fn decommit_to(self, sid: SessionId, party: PartyId) -> Result<(), _>;
    async fn recv_from(self, sid: SessionId, party: PartyId) -> Result<Vec<u8>, CheatDetectedError>;
}
```
In order for party A to commit to party B with new session id `sid`, party A calls `commit_to` and party B calls `expect_from` with corresponding arguments.
In general this operation is fallible, but the details of this failure are not actionable by the caller such as a network connection being dropped.
Once a pair of parties have completed the commitment for `sid`, the parties call `decommit_to` and `recv_from` respectively.
Unlike the commitment case where failure is unexpected, the party receiving a decommitment may detect that A tried to open the commitment to a new value and will want to report more information about how the cheat manifested.

Imagine we have a protocol `FooRand` that realizes `Frand` in the `Fcom` hybrid model, then it is expected that a `FooRandPlayer<FC>` would be generic over `FC: AsyncCom` and implement `AsyncRand`.
Some functionalities may work over multiple types of objects, such as `Fcote` that allows for correlated OT extension over an arbitrary ring `T` by providing `async fn send<T: Ring>(...)`.
Likewise some functionalities `AsyncBar<T>` would be generic over `T: Ring`, but a protocol may specialize to only implement over one ring `AsyncBar<RR2_128>`.


## Provided Functionalities

Currently the following functionalities are implemented by the library
- `Fnet` with `AsyncNetMgr` 
- `Fcom` with `FolkloreComPlayer` 
- `Frand` with `FolkloreRandPlayer`
- `Fcote` with `KosCotePlayer` that builds upon the [KOS15](https://eprint.iacr.org/2015/546) correlated OT protocol implemented in the EMP-OT library to support per-message correlations
- `Fmult` with `DklsMultPlayer` that implements the multiparty multiplication scheme of [DKLs19](https://eprint.iacr.org/2019/523)
- `Fmpc` with `WrkMpcPlayer` that utilizes the [WRK17](https://eprint.iacr.org/2017/189) scheme implemented by the EMP-agmpc library
- `Fabit` with `WrkAbitPlayer` that utilizes the [WRK17](https://eprint.iacr.org/2017/189) scheme implemented by the EMP-agmpc library
- `Ftabit` with `RstTabitPlayer`
- `Fthresh` with `RstThreshPlayer` and `GenericThreshPlayer`

## Questions
Please email rosefield.s (@) northeastern.edu
