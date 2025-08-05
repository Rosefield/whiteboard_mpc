use crate::{
    base_func::{BaseFunc, FuncId, SessionId},
    party::PartyId,
    //serialize::ToFromBytes,
};

use std::{
    collections::HashMap,
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tokio::{
    io::{self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    sync::{Mutex, Notify},
};

use log::trace;

struct RecvState<I>(Option<(SessionId, usize)>, HashMap<SessionId, Arc<Notify>>, I);

impl<I> RecvState<I> {
    fn new(i: I) -> Self {
        Self(None, HashMap::new(), i)
    }
}



pub struct AsyncNetworkMgr<I, O> {
    party_id: PartyId,
    recvs: HashMap<(PartyId, FuncId), Mutex<RecvState<I>>>,
    sends: HashMap<(PartyId, FuncId), Mutex<O>>,
    net_bytes: HashMap<(PartyId, FuncId), AtomicU64>,
}

impl<I, O> BaseFunc for AsyncNetworkMgr<I, O> {
    const FUNC_ID: FuncId = FuncId::Fnet;
    const REQUIRED_FUNCS: &'static [FuncId] = &[];

    fn party(&self) -> PartyId {
        self.party_id
    }
}

/// The trait that represents the network for the protocol.
/// It is responsible for delivering messages to other parties and named sub-components.
///
/// This trait allows for multiple concurrent and parallel sends and receives
pub trait AsyncNet: Send + Sync + 'static {
    /// Sends a message to (`party`, `func`, `sid`)
    fn send_to<B: AsRef<[u8]> + Send>(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        data: B,
    ) -> impl Future<Output = io::Result<()>> + Send;

    /// Receive a message from (`party`, `func`, `sid`),
    /// returning the buffer and the number of bytes received if successful.
    fn recv_from(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        buf: Arc<[u8]>,
    ) -> impl Future<Output = io::Result<(Arc<[u8]>, usize)>> + Send;

    /// Sends a message to (`party`, `func`, `sid`)
    async fn send_to_local<B: AsRef<[u8]>>(
        self: &Self,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        data: B,
    ) -> io::Result<()>;

    /// Receives a message from (`party`, `func`, `sid)
    async fn recv_from_local<B: AsMut<[u8]>>(
        self: &Self,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        buf: B,
    ) -> io::Result<(B, usize)>;

    fn reset_stats(self: &Self) -> HashMap<(PartyId, FuncId), u64>;

    /*
    /// Send a message to (`party`, `func`), but multiple bufs
    fn send_to_multi(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        bufs: MultiBuf,
    ) -> impl Future<Output = io::Result<MultiBuf>> + Send;

    /// Receive a message from (`party`, `func`), but writes into bufs
    fn recv_from_multi(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        bufs: MultiBuf,
    ) -> impl Future<Output = io::Result<(MultiBuf, usize)>> + Send;
    */
}

impl<I: AsyncRead + Unpin + Send + 'static, O: AsyncWrite + Unpin + Send + 'static> AsyncNet
    for AsyncNetworkMgr<I, O>
{
    async fn send_to<B: AsRef<[u8]> + Send>(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        data: B,
    ) -> io::Result<()> {
        self.send_to_local(party, func, sid, data).await
    }

    async fn recv_from(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        mut buf: Arc<[u8]>,
    ) -> io::Result<(Arc<[u8]>, usize)> {
        let b = Arc::get_mut(&mut buf).unwrap();

        let (_, s) = self.recv_from_local(party, func, sid, b).await?;

        Ok((buf, s))
    }

    async fn send_to_local<B: AsRef<[u8]>>(
        self: &Self,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        data: B,
    ) -> io::Result<()> {
        let data = data.as_ref();

        trace!(
            "{}: send to ({:?}, {}, {:?}) size {}",
            self.party_id,
            func,
            party,
            sid,
            data.len()
        );

        let mut target = self.sends[&(party, func)].lock().await;

        self.net_bytes[&(party, func)].fetch_add(data.len() as u64, Ordering::SeqCst);

        let _ = target.write(&(data.len() as u32).to_le_bytes()).await?;
        let _ = target.write(&sid.as_bytes()).await?;
        let _ = target.write(data).await?;
        target.flush().await?;

        Ok(())
    }

    async fn recv_from_local<B: AsMut<[u8]>>(
        self: &Self,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        buf: B,
    ) -> io::Result<(B, usize)> {
        self.try_recv_from_sid(party, func, sid, buf).await
    }

    fn reset_stats(self: &Self) -> HashMap<(PartyId, FuncId), u64> {
        // retrieve and reset stats
        self.net_bytes
            .iter()
            .map(|(k, v)| (k.clone(), v.swap(0, Ordering::SeqCst)))
            .collect()
    }


    /*
    /// Send a message to (`party`, `func`), but with multiple bufs
    async fn send_to_multi(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        bufs: MultiBuf,
    ) -> io::Result<MultiBuf> {
        let mut target = self.sends[&(party, func)].lock().await;

        let total_size: usize = bufs.total_size();
        let _ = target.write(&(total_size as u32).to_le_bytes()).await?;
        for b in bufs.iter() {
            let _ = target.write(b).await?;
        }
        target.flush().await?;

        Ok(bufs)
    }

    /// Receive a message from (`party`, `func`), but writes into bufs if necessary
    async fn recv_from_multi(
        self: Arc<Self>,
        party: PartyId,
        func: FuncId,
        mut bufs: MultiBuf,
    ) -> io::Result<(MultiBuf, usize)> {
        let mut other = self.recvs[&(party, func)].lock().await;

        let mut lb = [0u8; 4];
        other.read(&mut lb).await?;
        let size: usize = u32::from_le_bytes(lb).try_into().unwrap();

        // for now
        let avail_size = bufs.total_size();
        assert!(size <= avail_size);

        let mut remaining = size;
        while remaining > 0 {
            let b = bufs.next_buf_mut().unwrap();
            // fill each of the bufs in order, with the last buf partially filled
            let r = std::cmp::min(b.len(), remaining);
            other.read_exact(&mut b[..r]).await?;
            remaining -= r;
        }

        Ok((bufs, size))
    }
    */
}

impl<I: AsyncRead + Unpin + Send + 'static, O: AsyncWrite + Unpin + Send + 'static> AsyncNetworkMgr<I, O> {
    pub fn new(
        party_id: PartyId,
        _num_parties: usize,
        senders: HashMap<(PartyId, FuncId), O>,
        receivers: HashMap<(PartyId, FuncId), I>,
    ) -> Result<Self, ()> {
        let net_bytes = senders
            .keys()
            .map(|k| (k.clone(), AtomicU64::new(0)))
            .collect();

        Ok(AsyncNetworkMgr {
            party_id: party_id,
            sends: senders
                .into_iter()
                .map(|(k, v)| (k, Mutex::new(v)))
                .collect(),
            recvs: receivers
                .into_iter()
                .map(|(k, v)| (k, Mutex::new(RecvState::new(v))))
                .collect(),
            net_bytes: net_bytes,
        })
    }

    async fn try_recv_from_sid<B: AsMut<[u8]>>(&self,
        party: PartyId,
        func: FuncId,
        sid: SessionId,
        buf: B) -> io::Result<(B, usize)>  {

        let read_header = async |r: &mut I| -> io::Result<(usize, SessionId)> {
            let mut header = [0u8; 14];
            r.read_exact(&mut header).await?;
            //trace!("received header from ({party}, {func:?}): {header:?}");
            let (h1, h2) = header.split_first_chunk::<4>().unwrap();
            let size: usize = u32::from_le_bytes(*h1) as usize;
            let o_sid = SessionId::from_bytes(h2);
            Ok((size, o_sid))
        };


        let read_data = async |r: &mut I, mut buf: B, size: usize| -> io::Result<(B, usize)> {
            let b = buf.as_mut();
            trace!(
                "{}: recv from ({:?}, {}, {:?}), size {}/ buf {}",
                self.party_id,
                func,
                party,
                sid,
                size,
                b.len(),
            );

            // for now
            assert!(
                size <= b.len(),
                "self = {}, other = {party}, func = {func:?}, size = {size}, buf = {}",
                self.party_id,
                b.len()
            );

            self.net_bytes[&(party, func)].fetch_add(size as u64, Ordering::SeqCst);

            r.read_exact(&mut b[..size]).await?;

            Ok((buf, size))
        };

        let read_and_notify = async |mut guard: tokio::sync::MutexGuard<'_, RecvState<I>>, buf, sid, size| -> io::Result<(B, usize)>  {
                trace!(
                    "{}: recv from ({:?}, {}, {:?}) found sid header",
                    self.party_id,
                    func,
                    party,
                    sid,
                );

                // our message is next on the queue
                // since the size is set we were either notified previously or this is the
                // first call and so we don't need to wait on the notified

                // appease the borrow checker
                let (buf, read) = read_data( &mut guard.2, buf, size).await?;

                // we consumed the message
                guard.0 = None;
                // remove the notifier if it exists
                guard.1.remove(&sid);

                // If there was something in the queue that means there is someone else waiting
                // wake someone else so they can try to receive their message
                if let Some(n) = guard.1.values().next() {
                    n.notify_one();
                }

                return Ok((buf, read));
        };

        loop {
            let mut guard = self.recvs[&(party, func)].lock().await;

            // someone read the header for us
            if let Some((x, size)) = guard.0  {
                if x == sid {
                    return read_and_notify(guard, buf, sid, size).await;
                } 

                // get or add our notifier
                let ours = guard.1.entry(sid).or_insert_with(|| Arc::new(Notify::new()));
                let notify = ours.clone();
                // let someone else try
                drop(guard);

                // wait for someone to wake us again
                notify.notified().await;
            } else {
                // no header 
                let (size, o_sid) = read_header(&mut guard.2).await?;

                // found our message
                if sid == o_sid {
                    return read_and_notify(guard, buf, sid, size).await;
                } else {
                    // found a message for someone else
                    trace!("({}, {sid} : read header for other sid {o_sid}", self.party_id);
                    guard.0 = Some((o_sid, size));

                    // get or add our notifier
                    let ours = guard.1.entry(sid).or_insert_with(|| Arc::new(Notify::new()));
                    let notify = ours.clone();

                    // let the other sid waiting for our message know theirs is
                    let e = guard.1.entry(o_sid).or_insert_with(|| Arc::new(Notify::new()));
                    e.notify_one();
                    
                    // let someone else try
                    drop(guard);

                    // wait for someone to wake us again
                    notify.notified().await;
                }
            }
        }
    }

}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::party::{PartyId, PartyInfo};
    use std::{net::IpAddr, str::FromStr};
    use tokio::{
        net::{TcpListener, TcpStream},
        io::{simplex, SimplexStream, ReadHalf, WriteHalf},
        task::JoinSet,
    };
    use rand::seq::SliceRandom;

    pub fn get_test_party_infos(num: PartyId) -> Vec<PartyInfo> {
        (1..=num)
            .map(|p| PartyInfo {
                id: p,
                ip: IpAddr::from_str(&format!("127.0.0.1")).unwrap(),
                port: 9000 + 1000 * p,
            })
            .collect()
    }

    pub async fn build_test_nets_tcp(
        party_info: &[PartyInfo],
        funcs: Vec<FuncId>,
    ) -> Vec<Arc<AsyncNetworkMgr<TcpStream, TcpStream>>> {
        let mut nets = Vec::new();

        let mut senders: HashMap<PartyId, HashMap<(PartyId, FuncId), TcpStream>> = HashMap::new();
        let mut receivers: HashMap<PartyId, HashMap<(PartyId, FuncId), TcpStream>> = HashMap::new();

        for pi in party_info.iter() {
            for pj in party_info.iter().filter(|pj| pj.id != pi.id) {
                for &f in funcs.iter() {
                    // ask for a new port
                    let address = (pi.ip.clone(), 0);
                    let listener = TcpListener::bind(address.clone()).await.unwrap();
                    let actual_address = listener.local_addr().unwrap();
                    let hs = tokio::spawn(async move { listener.accept().await.unwrap() });
                    let hr =
                        tokio::spawn(
                            async move { TcpStream::connect(actual_address).await.unwrap() },
                        );

                    let (s, _) = hs.await.unwrap();
                    let r = hr.await.unwrap();

                    senders
                        .entry(pi.id)
                        .or_insert(HashMap::new())
                        .insert((pj.id, f), s);
                    receivers
                        .entry(pj.id)
                        .or_insert(HashMap::new())
                        .insert((pi.id, f), r);
                }
            }
        }

        for pi in party_info.iter() {
            let n = AsyncNetworkMgr::new(
                pi.id,
                party_info.len(),
                senders.remove(&pi.id).unwrap(),
                receivers.remove(&pi.id).unwrap(),
            );
            nets.push(Arc::new(n.unwrap()));
        }

        nets
    }

    pub async fn build_test_nets(
        party_info: &[PartyInfo],
        funcs: Vec<FuncId>,
    ) -> Vec<Arc<AsyncNetworkMgr<ReadHalf<SimplexStream>, WriteHalf<SimplexStream>>>> {
        let mut nets = Vec::new();

        let mut senders: HashMap<PartyId, HashMap<(PartyId, FuncId), WriteHalf<_>>> = HashMap::new();
        let mut receivers: HashMap<PartyId, HashMap<(PartyId, FuncId), ReadHalf<_>>> = HashMap::new();

        for pi in party_info.iter() {
            for pj in party_info.iter().filter(|pj| pj.id != pi.id) {
                for &f in funcs.iter() {
                    // probably could just use a simplex stream here 
                    let (r, s) = simplex(1 << 20);

                    senders
                        .entry(pi.id)
                        .or_insert(HashMap::new())
                        .insert((pj.id, f), s);
                    receivers
                        .entry(pj.id)
                        .or_insert(HashMap::new())
                        .insert((pi.id, f), r);
                }
            }
        }

        for pi in party_info.iter() {
            let n = AsyncNetworkMgr::new(
                pi.id,
                party_info.len(),
                senders.remove(&pi.id).unwrap(),
                receivers.remove(&pi.id).unwrap(),
            );
            nets.push(Arc::new(n.unwrap()));
        }

        nets
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn send_recv() -> io::Result<()> {
        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Ftest];
        let nets = build_test_nets(&party_info, funcs).await;

        let net1 = nets[0].clone();
        let net2 = nets[1].clone();

        let sid = SessionId::new(FuncId::Ftest);


        let h1 = tokio::spawn(async move {
            let r = net1
                .send_to(2, FuncId::Ftest, sid, Arc::from([1, 2, 3, 4].as_slice()))
                .await;
            assert!(r.is_ok());
        });

        let h2 = tokio::spawn(async move {
            let buf = Arc::from([0; 4]);
            let r = net2.recv_from(1, FuncId::Ftest, sid, buf).await;
            assert!(r.is_ok());
            let (b, _) = r.unwrap();
            assert!(*b == [1, 2, 3, 4]);
        });

        h1.await?;
        h2.await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_out_of_order_sid_concurrency() -> io::Result<()> {

        let _ = env_logger::builder().is_test(true).try_init();

        let party_info = get_test_party_infos(2);
        let funcs = vec![FuncId::Ftest];
        let nets = build_test_nets(&party_info, funcs).await;

        let net1 = nets[0].clone();
        let net2 = nets[1].clone();

        let num = 1000u64;

        let h1 = tokio::spawn(async move {

            let ids = {
                let mut rng = rand::thread_rng();
                let mut vec: Vec<_> = (0..num).collect();
                vec.shuffle(&mut rng);

                vec
            };
            
            for i in ids {
                let sid = SessionId {
                    parent: FuncId::Ftest,
                    id: i
                };
                let _ = net1
                .send_to_local(2, FuncId::Ftest, sid, sid.as_bytes()).await;
            }
        });

        let h2 = tokio::spawn(async move {

            // with known in-order messages (e.g. no contention) this takes 2ms
            // compared to having random order (+ spawned receivers) this takes 10-14ms
            let ids: Vec<_> = (0..num).collect();

            let now = std::time::Instant::now();


            /*
            // in order
            for i in ids {
                let sid = SessionId {
                    parent: FuncId::Ftest,
                    id: i
                };
                let buf = [0u8; 10];
                let res = net2
                    .recv_from_local(1, FuncId::Ftest, sid, buf).await;
                assert!(res.is_ok());
                let (b, _) = res.unwrap();
                assert_eq!(sid.as_bytes(), b);
            }
            */

            // out of order
            let mut set = JoinSet::new();
            for i in ids {
                set.spawn({
                    let net2 = net2.clone();

                    async move {
                    let sid = SessionId {
                        parent: FuncId::Ftest,
                        id: i
                    };
                    let buf = [0u8; 10];
                    let res = net2
                    .recv_from_local(1, FuncId::Ftest, sid, buf).await;
                    (sid, res)
                }});
            }

            let recvs = set.join_all().await;

            for (sid, res) in recvs {
                assert!(res.is_ok());
                let (b, _) = res.unwrap();
                assert_eq!(sid.as_bytes(), b);
            }

            println!("Recv all took {}ms", now.elapsed().as_millis());
        });

        let (r1, r2) = tokio::join!(h1, h2);
        assert!(r1.is_ok() && r2.is_ok());

        Ok(())
    }

}
