use thresh_mpc::{
    base_func::{FuncId, SessionId, UnexpectedError},
    circuits::{
        aes::{aes_circuit, aes_key_schedule, aes_with_schedule},
        arith::sum_circuit,
        elements::hardcode_input,
        keccak::{kmac_128, kmac_128_post, kmac_128_pre},
        new_builder,
        sha256::{hmacsha256_circuit, hmacsha256_post_circuit, hmacsha256_pre_circuit},
        CircuitElement, CircuitRing, Gate, TCircuit,
    },
    common_protos::synchronize,
    ff2_128::FF2_128,
    func_abit::WrkAbitPlayer,
    func_com::FolkloreComPlayer,
    func_cote::KosCotePlayer,
    func_mpc::{AsyncMpc, WrkMpcPlayer},
    func_mult::DklsMultPlayer,
    func_net::{AsyncNet, AsyncNetworkMgr},
    func_rand::FolkloreRandPlayer,
    func_thresh::{AsyncThresh, GenericThreshPlayer, RstThreshPlayer},
    func_thresh_abit::RstTabitPlayer,
    party::{PartyId, PartyInfo},
    utils::parse_party_file,
};

use std::{collections::HashMap, sync::Arc, time::Instant};

use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    task::JoinSet,
    time::{sleep, Duration},
};

use argh::FromArgs;

use rand::Rng;

use env_logger::{Builder, Target};
use log::{info, warn};

#[derive(FromArgs)]
/// Configuration information for running the protocol
struct RunInformation {
    #[argh(option, short = 'm')]
    /// the ID of this party
    my_id: PartyId,

    #[argh(option, short = 't')]
    /// the threshold parameter for number of corrupt parties
    threshold: u16,

    #[argh(option, short = 'e')]
    /// the name of the example to run
    example: Option<String>,

    #[argh(option, short = 's')]
    /// a file to save/load state from
    state_file: Option<String>,

    #[argh(option, short = 'p')]
    /// name of the file that gives the connection information for each party
    party_file: String,

    #[argh(switch)]
    /// only perform the setup computation
    only_setup: bool,

    #[argh(switch)]
    /// use the generic thresh as the thresh implementation
    use_generic: bool,
}

async fn make_party_connections(
    my_id: PartyId,
    party_info: &Vec<PartyInfo>,
    funcs: &[FuncId],
) -> (
    HashMap<(PartyId, FuncId), impl AsyncWrite>,
    HashMap<(PartyId, FuncId), impl AsyncRead>,
) {
    let mut senders = HashMap::new();
    let mut receivers = HashMap::new();

    let my_info = party_info.iter().find(|x| x.id == my_id).unwrap();

    let mut sends = JoinSet::new();
    let mut recvs = JoinSet::new();

    for (f, &_f) in funcs.iter().enumerate() {
        let port = my_info.port + (f as u16);
        let listener = Arc::new(TcpListener::bind((my_info.ip, port)).await.unwrap());
        for pj in party_info.iter().cloned().filter(|x| x.id != my_id) {
            let l = listener.clone();
            sends.spawn(async move {
                let (mut stream, _) = l.accept().await.unwrap();
                let other = stream.read_u16().await.unwrap();

                ((other, _f), BufWriter::new(stream))
            });
            let their_port = pj.port + (f as u16);
            recvs.spawn(async move {
                for _ in 0..10 {
                    if let Ok(mut stream) = TcpStream::connect((pj.ip, their_port)).await {
                        stream.write_u16(my_id).await.unwrap();

                        return ((pj.id, _f), BufReader::new(stream));
                    };
                    sleep(Duration::from_millis(1000)).await;
                }

                panic!("failed to connect to {}", pj.ip);
            });
        }
    }

    while let Some(t) = sends.join_next().await {
        let (k, v) = t.unwrap();
        senders.insert(k, v);
    }

    while let Some(t) = recvs.join_next().await {
        let (k, v) = t.unwrap();
        receivers.insert(k, v);
    }

    (senders, receivers)
}

type F = FF2_128;

async fn setup_systems(
    info: &RunInformation,
) -> Result<
    (
        Vec<PartyId>,
        Arc<impl AsyncNet>,
        Arc<WrkMpcPlayer<F>>,
        impl AsyncThresh,
        impl AsyncThresh,
    ),
    (),
> {
    // read other party information
    let (num_parties, party_info) = parse_party_file(&info.party_file);
    let all_parties: Vec<PartyId> = party_info.iter().map(|x| x.id.clone()).collect();
    let threshold = info.threshold;
    let stat_param = 80;
    let net_funcs = [
        FuncId::Fcom,
        FuncId::Fcote,
        FuncId::Fmult,
        FuncId::Ftabit,
        FuncId::Fcontroller,
    ];

    let (sends, recvs) = make_party_connections(info.my_id, &party_info, &net_funcs).await;

    let net: Arc<AsyncNetworkMgr<_, _>> = Arc::new(AsyncNetworkMgr::new(
        info.my_id,
        num_parties.into(),
        sends,
        recvs,
    )?);
    let com: Arc<FolkloreComPlayer<_>> = Arc::new(FolkloreComPlayer::new(
        info.my_id,
        num_parties,
        net.clone(),
    )?);
    let rand: Arc<FolkloreRandPlayer<_>> = Arc::new(FolkloreRandPlayer::new(
        info.my_id,
        num_parties,
        com.clone(),
    )?);
    let cote: Arc<KosCotePlayer<_>> =
        Arc::new(KosCotePlayer::new(info.my_id, &party_info, net.clone())?);
    let mult: Arc<DklsMultPlayer<F, _, _>> = Arc::new(DklsMultPlayer::new(
        info.my_id,
        num_parties.into(),
        stat_param,
        net.clone(),
        cote,
    )?);
    let abit: Arc<WrkAbitPlayer<F>> =
        Arc::new(WrkAbitPlayer::new(info.my_id, num_parties, &party_info)?);
    let tabit: Arc<RstTabitPlayer<F, _, _, _, _, _>> = Arc::new(RstTabitPlayer::new(
        info.my_id,
        num_parties,
        threshold,
        abit,
        rand,
        mult,
        com.clone(),
        net.clone(),
    )?);
    let mpc: Arc<WrkMpcPlayer<F>> = Arc::new(WrkMpcPlayer::new(info.my_id, &party_info)?);
    let comcomp = GenericThreshPlayer::new(info.my_id, num_parties, threshold, mpc.clone())?;
    let thresh: RstThreshPlayer<F, _, _> =
        RstThreshPlayer::new(info.my_id, num_parties, threshold, mpc.clone(), tabit)?;

    //Ok((all_parties, net, mpc, thresh, comcomp))
    Ok((all_parties, net, mpc, comcomp, thresh))
}

fn setup_circuit<I: CircuitRing, O>(np: usize, cir: &TCircuit<I, O>) -> TCircuit<Vec<I>, (I, O)> {
    let b = new_builder();
    let (b, ids) = b.add_input_multi::<Vec<I>>(np, None);
    let sum = sum_circuit::<I>(np);
    let (b, k) = b.extend_circuit(&ids, &sum, None);
    let (b, mut keys) = b.extend_circuit(&k, cir, None);
    let mut outs = k;
    outs.append(&mut keys);

    b.refine_input().refine_output(&outs).to_circuit()
}

async fn run_setup<I: CircuitRing, O: CircuitElement>(
    info: &RunInformation,
    parties: &[PartyId],
    net: Arc<impl AsyncNet>,
    mpc: &impl AsyncMpc<F>,
    //comcomp: &impl AsyncThresh,
    thresh: &mut impl AsyncThresh,
    pre_circuit: &TCircuit<I, O>,
    name: &str,
    ids: &[usize],
    mut mpc_sid: SessionId,
) -> Result<SessionId, UnexpectedError> {
    let np = parties.len();

    let sid = SessionId::new(FuncId::Fcontroller);

    synchronize(info.my_id, parties, FuncId::Fcontroller, sid, net.clone()).await?;
    let ins = pre_circuit.inputs.len();
    let input = {
        let mut input = vec![false; ins];
        let mut rng = rand::thread_rng();
        rng.fill(&mut input[..]);
        input
    };

    {
        let circuit = setup_circuit(np, pre_circuit);
        let start = Instant::now();
        info!("{}: Running setup with thresh {}", info.my_id, name);
        let _ = thresh.setup(&input, ids, &circuit).await?;

        info!(
            "{}: Setup {} complete in {:?}",
            info.my_id,
            name,
            start.elapsed()
        );

        let stats = net.reset_stats();
        let net_traffic: u64 = stats.into_values().sum();
        info!(
            "{}: Setup {} other net traffic {:?}",
            info.my_id, name, net_traffic
        );
    }

    synchronize(info.my_id, parties, FuncId::Fcontroller, sid, net.clone()).await?;

    {
        info!("{}: Running setup with plain {}", info.my_id, name);
        let _ = mpc.init(mpc_sid, None).await?;
        let input = if info.my_id == 1 { Some(input) } else { None };
        let _ = mpc.input_multi(mpc_sid, 1, ins, input).await?;
        let _ = mpc.eval_pub(mpc_sid, &parties, pre_circuit).await?;
        mpc_sid = mpc_sid.next();
    }
    Ok(mpc_sid)
}

async fn run_example<const N: usize, I, O2: CircuitElement>(
    info: &RunInformation,
    parties: &[PartyId],
    net: Arc<impl AsyncNet>,
    mpc: &impl AsyncMpc<F>,
    //comcomp: &impl AsyncThresh,
    thresh: &mut impl AsyncThresh,
    full_circuit: &TCircuit<I, [bool; N]>,
    post_circuit: &TCircuit<O2, [bool; N]>,
    name: &str,
    ids: &[usize],
    mut mpc_sid: SessionId,
) -> Result<SessionId, UnexpectedError> {
    let s = |bits: &[bool]| {
        let b: Vec<_> = bits
            .iter()
            .map(|bit| if *bit { "1" } else { "0" })
            .collect();
        b.join("")
    };

    let sid = SessionId::new(FuncId::Fcontroller).next();

    synchronize(info.my_id, parties, FuncId::Fcontroller, sid, net.clone()).await?;

    {
        let circuit = post_circuit;
        let start = Instant::now();
        info!(
            "{}: Running with {:?} thresh {} with preprocessing",
            info.my_id, parties, name
        );
        let r = thresh.eval(parties, ids, circuit).await?;
        info!(
            "{}: thresh {} complete in {:?}",
            info.my_id,
            name,
            start.elapsed()
        );
        info!("{}: {name}_k(0) = {}", info.my_id, s(&r));
    }

    synchronize(info.my_id, parties, FuncId::Fcontroller, sid, net.clone()).await?;

    {
        let ins = full_circuit.inputs.len();
        let input = vec![false; ins];
        info!(
            "{}: Running with {:?} plain {} full circuit",
            info.my_id, parties, name
        );
        let _ = mpc.init(mpc_sid, None).await?;
        let input = if info.my_id == 1 { Some(input) } else { None };
        let _ = mpc.input_multi(mpc_sid, 1, ins, input).await?;
        let _ = mpc.eval_pub(mpc_sid, parties, full_circuit).await?;
        mpc_sid = mpc_sid.next();
    }
    Ok(mpc_sid)
}

fn describe<I, O>(name: &str, c: &TCircuit<I, O>) {
    let num_ands = |gs: &[Gate]| -> usize {
        gs.iter()
            .map(|g| match g {
                Gate::And(_, _, _) => 1usize,
                _ => 0,
            })
            .sum()
    };

    info!(
        "{name} size (|I| = {}, |G| = {}, |O| = {})",
        c.inputs.len(),
        num_ands(&c.gates),
        c.outputs.len()
    );
}

async fn run_all(
    mut thresh: impl AsyncThresh,
    info: RunInformation,
    net: Arc<impl AsyncNet>,
    mpc: Arc<WrkMpcPlayer<F>>,
    all_parties: Vec<PartyId>,
) -> Result<(), UnexpectedError> {
    let nparties = all_parties.len();

    let aes_pt = [false; 128];
    let aes = hardcode_input(aes_pt, &aes_circuit());
    let aes_schedule = aes_key_schedule();
    let aes_expanded = hardcode_input(aes_pt, &aes_with_schedule());

    let hmac_pt = [false; 256];
    let hmac = hardcode_input(hmac_pt, &hmacsha256_circuit());
    let hmac_pre = hmacsha256_pre_circuit();
    let hmac_post = hardcode_input(hmac_pt, &hmacsha256_post_circuit());

    let kmac_pt = [false; 256];
    let kmac = hardcode_input(kmac_pt, &kmac_128());
    let kmac_pre = kmac_128_pre();
    let kmac_post = hardcode_input(kmac_pt, &kmac_128_post());

    // first 128 wires are the key, remaining 1280 are the rest of the schedule
    let key_schedule_ids: Vec<_> = (1..=1408).collect();
    // first 256 wires is the shared key, remaining 512 the sha chaining-states
    let hmac_pre_ids: Vec<_> = (2000..2768).collect();

    let kmac_pre_ids: Vec<_> = (3000..4856).collect();

    let mut mpc_sid = SessionId::new(FuncId::Fcontroller);

    let mut needs_init = true;

    if let Some(state_file) = info.state_file.as_ref() {
        let r = thresh.resume_from_state_file(&state_file, false).await;
        if r.is_err() {
            warn!(
                "{}: expected to resume from {} but failed ({:?}), rerunning setup",
                info.my_id, state_file, r
            );
        } else {
            needs_init = false;
        }
    }

    if needs_init {
        info!("Running init");
        // pre-run MPC init to make sure network is created
        mpc.init(mpc_sid, None).await?;
        mpc_sid = mpc_sid.next();

        thresh.init().await?;

        info!("Running preprocessing: ");
        describe("AES-KeySchedule", &aes_schedule);
        mpc_sid = run_setup(
            &info,
            &all_parties,
            net.clone(),
            mpc.as_ref(),
            &mut thresh,
            &aes_schedule,
            "AES-KeySchedule",
            &key_schedule_ids,
            mpc_sid,
        )
        .await?;

        describe("HMAC-SHA256 Chain", &hmac_pre);
        mpc_sid = run_setup(
            &info,
            &all_parties,
            net.clone(),
            mpc.as_ref(),
            &mut thresh,
            &hmac_pre,
            "HMAC-SHA256 Chaining State",
            &hmac_pre_ids,
            mpc_sid,
        )
        .await?;

        describe("KMAC-128 Initial State", &kmac_pre);
        mpc_sid = run_setup(
            &info,
            &all_parties,
            net.clone(),
            mpc.as_ref(),
            &mut thresh,
            &kmac_pre,
            "KMAC-128 Initial State",
            &kmac_pre_ids,
            mpc_sid,
        )
        .await?;

        if let Some(state_file) = info.state_file.as_ref() {
            thresh
                .write_state_to_file(&state_file)
                .expect("failed to write to state file");
        }
    }

    if info.only_setup {
        return Ok(());
    }

    info!("Running with test cases: ");
    describe("AES", &aes);
    describe("AES-Expanded", &aes_expanded);

    describe("HMAC-SHA256", &hmac);
    describe("HMAC-SHA256 post", &hmac_post);

    describe("KMAC", &kmac);
    describe("KMAC with State", &kmac_post);

    let num_party_cases = vec![nparties]; //vec![8, 5, 3];

    let nruns = 1;

    // run with the most parties first so that when parties are done they can just exit
    for np in num_party_cases.into_iter() {
        let parties = all_parties[..np].to_vec();
        if !parties.contains(&info.my_id) {
            break;
        }

        for _ in 0..nruns {
            mpc_sid = run_example(
                &info,
                &parties,
                net.clone(),
                mpc.as_ref(),
                &mut thresh,
                &aes,
                &aes_expanded,
                "AES",
                &key_schedule_ids,
                mpc_sid,
            )
            .await?;
        }

        mpc_sid = run_example(
            &info,
            &parties,
            net.clone(),
            mpc.as_ref(),
            &mut thresh,
            &hmac,
            &hmac_post,
            "HMAC-SHA256",
            &hmac_pre_ids[256..],
            mpc_sid,
        )
        .await?;

        mpc_sid = run_example(
            &info,
            &parties,
            net.clone(),
            mpc.as_ref(),
            &mut thresh,
            &kmac,
            &kmac_post,
            "KMAC",
            &kmac_pre_ids[256..],
            mpc_sid,
        )
        .await?;
    }

    /*
    let stats = mpc.get_execution_stats();

    println!("{:?}", stats);
    */

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), UnexpectedError> {
    Builder::from_default_env().target(Target::Stdout).init();

    let info: RunInformation = argh::from_env();

    let (all_parties, net, mpc, comcomp, thresh) = setup_systems(&info).await.unwrap();

    if info.use_generic {
        run_all(comcomp, info, net, mpc, all_parties).await
    } else {
        run_all(thresh, info, net, mpc, all_parties).await
    }
}
