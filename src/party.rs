use std::net::IpAddr;

pub type PartyId = u16;

#[derive(Copy, Clone, Debug)]
pub struct PartyInfo {
    pub id: PartyId,
    pub ip: IpAddr,
    pub port: u16,
}

/*
impl PartyId {
    fn am_i_min_party(&self) -> bool {
        self == 1
    }
}
*/
