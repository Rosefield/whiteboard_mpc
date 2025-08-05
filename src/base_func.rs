use crate::party::PartyId;

use std::hash::Hasher;

#[derive(PartialEq, Copy, Clone, Eq, Hash, Debug)]
pub struct FuncId(u16);

#[allow(non_upper_case_globals)]
impl FuncId {
    pub const Fcomcomp: Self = Self(1);
    pub const Fcom: Self = Self(2);
    pub const Fmpc: Self = Self(3);
    pub const Fthresh: Self = Self(4);
    pub const Ftabit: Self = Self(5);
    pub const Frand: Self = Self(6);
    pub const Fzero: Self = Self(7);
    pub const Fmult: Self = Self(8);
    pub const Fabit: Self = Self(9);

    pub const Fot: Self = Self(100);
    pub const Feot: Self = Self(101);
    pub const Fote: Self = Self(102);
    pub const Fcote: Self = Self(103);
    pub const Fvole: Self = Self(104);

    pub const Fzk: Self = Self(200);

    pub const Fecdsa: Self = Self(300);

    pub const Fnet: Self = Self(999);
    pub const Ftest: Self = Self(1000);
    pub const Fcontroller: Self = Self(10000);
    pub const Other: Self = Self(65535);

    pub fn as_bytes(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }
}

impl From<u16> for FuncId {
    fn from(item: u16) -> Self {
        Self(item)
    }
}

impl From<FuncId> for u16 {
    fn from(item: FuncId) -> Self {
        item.0
    }
}

#[derive(PartialEq, Copy, Clone, Eq, Hash, Debug)]
pub struct SessionId {
    pub parent: FuncId,
    pub id: u64,
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({:?}, {})", self.parent, self.id)
    }
}

impl SessionId {
    pub fn new(caller: FuncId) -> Self {
        SessionId {
            parent: caller,
            id: 0,
        }
    }

    pub fn next(mut self) -> Self {
        self.id += 1;
        self
    }

    pub fn as_bytes(self) -> [u8; 10] {
        let mut bytes = [0; 10];
        bytes[0..2].copy_from_slice(&u16::to_le_bytes(self.parent.into()));
        bytes[2..10].copy_from_slice(&self.id.to_le_bytes());
        bytes
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let func = u16::from_le_bytes(b[..2].try_into().unwrap());
        let id = u64::from_le_bytes(b[2..10].try_into().unwrap());

        Self {
            parent: func.into(),
            id
        }
    }

    pub fn derive_ssid(&self, caller: FuncId) -> Self {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u16(self.parent.into());
        h.write_u64(self.id);
        // _probably_ collision free in our limited use case
        // use top 48 bits as the parent id, bottom 16 as counter
        let subid = h.finish() << 16;
        SessionId {
            parent: caller,
            id: subid,
        }
    }

    pub fn derive_ssid_context<T: std::hash::Hash>(&self, caller: FuncId, t: &T) -> Self {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u16(self.parent.into());
        h.write_u64(self.id);
        t.hash(&mut h);
        // _probably_ collision free in our limited use case
        // use top 48 bits as the parent id, bottom 16 as counter
        let subid = h.finish() << 16;
        SessionId {
            parent: caller,
            id: subid,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FuncContext {
    pub party: PartyId,
    pub func: FuncId,
    pub sid: SessionId,
}

impl std::fmt::Display for FuncContext {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} - {:?} {}", self.party, self.func, self.sid)
    }
}

pub trait BaseFunc {
    const FUNC_ID: FuncId;
    const REQUIRED_FUNCS: &'static [FuncId];

    fn party(&self) -> PartyId;

    fn ctx(&self, sid: SessionId) -> FuncContext {
        let c = FuncContext {
            party: self.party(),
            func: Self::FUNC_ID,
            sid: sid,
        };
        c
    }

    fn err(&self, sid: SessionId, msg: impl std::fmt::Display) -> String {
        let c = self.ctx(sid);
        format!("{c}: {msg}")
    }

    fn unexpected(&self, sid: SessionId, msg: impl std::fmt::Display) -> UnexpectedError {
        anyhow::anyhow!(self.err(sid, msg)).into()
    }

    fn cheat(&self, sid: SessionId, cheater: Option<PartyId>, msg: String) -> CheatDetectedError {
        let c = self.ctx(sid);

        CheatDetectedError::new(c, cheater, msg)
    }
}

#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct UnexpectedError(
    #[from]
    #[backtrace]
    anyhow::Error,
);

#[derive(thiserror::Error, Debug)]
#[error("{ctx}: Cheat detected by party {cheater:?}, {msg}")]
pub struct CheatDetectedError {
    ctx: FuncContext,
    cheater: Option<PartyId>,
    msg: String,
}

impl CheatDetectedError {
    pub fn new(ctx: FuncContext, cheater: Option<PartyId>, msg: String) -> Self {
        Self { ctx, cheater, msg }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CheatOrUnexpectedError {
    #[error(transparent)]
    CheatDetected(
        #[from]
        #[backtrace]
        CheatDetectedError,
    ),
    #[error(transparent)]
    Unexpected(
        #[from]
        #[backtrace]
        UnexpectedError,
    ),
}

impl From<std::io::Error> for CheatOrUnexpectedError {
    fn from(e: std::io::Error) -> Self {
        CheatOrUnexpectedError::Unexpected(UnexpectedError(e.into()))
    }
}

impl From<anyhow::Error> for CheatOrUnexpectedError {
    fn from(e: anyhow::Error) -> Self {
        CheatOrUnexpectedError::Unexpected(e.into())
    }
}

pub struct Assert<const B: bool>;
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_ssid_stability() {
        let sid = SessionId::new(FuncId::Ftest);

        let ssid = sid.derive_ssid(FuncId::Fcontroller);
        // within a program instance this should be deterministic
        assert_eq!(ssid, sid.derive_ssid(FuncId::Fcontroller));

        let sid2 = sid.clone().next();
        let ssid2 = sid2.derive_ssid(FuncId::Fcontroller);

        // different parents should result in different sub sids
        assert!(ssid != ssid2);
    }
}
