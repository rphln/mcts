#![feature(file_buffered)]

use std::{
    array,
    fs::File,
    io::{self, Write},
    num::ParseIntError,
    panic::catch_unwind,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use rand::prelude::*;
use swift_swallow::{
    Outcome, Rules, Subscribe, World,
    event::Start,
    grid::{Position, qr},
    rules::characters::{self, CharacterBuilder},
    world::Tile,
};
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

const TEAMS: [[usize; 4]; 15] = [
    [0, 1, 2, 3],
    [0, 1, 2, 4],
    [0, 1, 2, 5],
    [0, 1, 3, 4],
    [0, 1, 3, 5],
    [0, 1, 4, 5],
    [0, 2, 3, 4],
    [0, 2, 3, 5],
    [0, 2, 4, 5],
    [0, 3, 4, 5],
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
];

const POSITIONS: [[Position; 4]; 15] = [
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(4, 1)],
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    [qr(2, 0), qr(2, -1), qr(3, 0), qr(3, -1)],
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    [qr(2, 0), qr(4, -1), qr(3, 0), qr(3, -1)],
    [qr(2, 0), qr(4, 1), qr(3, 0), qr(3, -1)],
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    [qr(2, 0), qr(4, -1), qr(3, 0), qr(3, -1)],
    [qr(2, 0), qr(4, 1), qr(3, 0), qr(3, -1)],
    [qr(4, -1), qr(4, 1), qr(3, 0), qr(3, -1)],
];

// Ordered so the character mix stays even over time (an interrupted run is
// still balanced).
const PAIRS: [(usize, usize); 90] = [
    (0, 5),
    (1, 14),
    (9, 11),
    (4, 6),
    (12, 3),
    (14, 2),
    (7, 10),
    (13, 8),
    (1, 4),
    (0, 12),
    (5, 7),
    (14, 3),
    (2, 6),
    (9, 10),
    (13, 2),
    (0, 8),
    (1, 13),
    (11, 9),
    (6, 4),
    (7, 12),
    (5, 10),
    (3, 7),
    (12, 0),
    (14, 5),
    (1, 9),
    (3, 11),
    (11, 8),
    (8, 3),
    (4, 10),
    (9, 2),
    (12, 6),
    (7, 13),
    (10, 2),
    (0, 14),
    (4, 1),
    (8, 13),
    (5, 11),
    (14, 0),
    (6, 5),
    (12, 9),
    (11, 1),
    (0, 9),
    (3, 2),
    (12, 7),
    (3, 14),
    (8, 10),
    (4, 8),
    (10, 4),
    (6, 13),
    (7, 1),
    (2, 13),
    (5, 0),
    (11, 6),
    (9, 12),
    (14, 4),
    (1, 7),
    (13, 1),
    (2, 3),
    (10, 8),
    (4, 14),
    (6, 2),
    (13, 0),
    (5, 14),
    (7, 5),
    (6, 11),
    (10, 5),
    (8, 0),
    (11, 3),
    (4, 12),
    (9, 1),
    (13, 7),
    (1, 11),
    (10, 9),
    (2, 9),
    (6, 12),
    (2, 10),
    (8, 4),
    (13, 6),
    (3, 8),
    (11, 5),
    (9, 0),
    (2, 14),
    (10, 7),
    (3, 12),
    (7, 3),
    (8, 11),
    (0, 13),
    (5, 6),
    (12, 4),
    (14, 1),
];

/// Self-play balance harness.
#[derive(Debug, Parser)]
pub struct Args {
    /// Number of matches to play; ideally a multiple of 90.
    #[clap(long, default_value_t = 1080)]
    pub matches: usize,
    /// Worker threads.
    #[clap(long, default_value_t = 16)]
    pub threads: usize,
    /// MCTS time budget per move, in milliseconds.
    #[clap(long, value_parser = parse_millis, required_unless_present_any = ["iters", "nodes"])]
    pub time: Option<Duration>,
    /// MCTS iteration budget per move.
    #[clap(long, required_unless_present_any = ["time", "nodes"])]
    pub iters: Option<u32>,
    /// MCTS node budget per move.
    #[clap(long, required_unless_present_any = ["time", "iters"])]
    pub nodes: Option<usize>,
    /// Output CSV, rewritten after every match.
    #[clap(long, default_value = "matches.csv")]
    pub out: PathBuf,
}

fn parse_millis(arg: &str) -> Result<Duration, ParseIntError> {
    Ok(Duration::from_millis(arg.parse()?))
}

#[derive(Copy, Clone, Default)]
struct Record {
    wins: u32,
    draws: u32,
    losses: u32,
}

struct Shared {
    args: Args,
    next: AtomicUsize,
    done: AtomicUsize,
    tally: Mutex<[Record; 90]>,
    start: Instant,
}

fn main() {
    let args = Args::parse();
    let threads = args.threads;

    let shared = Arc::new(Shared {
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        tally: Mutex::new([Record::default(); 90]),
        start: Instant::now(),
        args,
    });

    let handles: Vec<_> = (0..threads)
        .map(|_worker| {
            let shared = shared.clone();
            thread::spawn(move || run_worker(&shared))
        })
        .collect();

    for handle in handles {
        if let Err(_reason) = handle.join() {
            eprintln!("run compromised: a match panicked");
            std::process::exit(1);
        }
    }
}

#[expect(clippy::cast_precision_loss)]
fn run_worker(shared: &Shared) {
    let characters = [
        characters::paladin(),
        characters::warrior(),
        characters::scholar(),
        characters::sidewinder(),
        characters::rogue(),
        characters::warlock(),
    ];

    loop {
        let n = shared.next.fetch_add(1, Ordering::Relaxed);
        if n >= shared.args.matches {
            break;
        }

        let seed = n as u64;
        let pair = n % PAIRS.len();

        let (white, black) = PAIRS[pair];

        let white_team: [CharacterBuilder; 4] = array::from_fn(|idx| {
            characters[TEAMS[white][idx]].clone().position(POSITIONS[white][idx])
        });
        let black_team: [CharacterBuilder; 4] = array::from_fn(|idx| {
            characters[TEAMS[black][idx]].clone().position(-POSITIONS[black][idx])
        });

        let outcome = catch_unwind(|| {
            let game = build_game(seed, &white_team, &black_team);
            let mut rng = DefaultRng::seed_from_u64(seed);
            play_game(
                game,
                &mut rng,
                shared.args.time,
                shared.args.iters,
                shared.args.nodes,
            )
        })
        .unwrap_or_else(|reason| {
            panic!("match panicked: n={n} seed={seed} white={white} black={black} reason={reason:?}")
        });

        let mut records = shared.tally.lock().unwrap();
        match outcome {
            Outcome::Draw => records[pair].draws += 1,
            Outcome::Victory(Color::White) => records[pair].wins += 1,
            Outcome::Victory(Color::Black) => records[pair].losses += 1,
        }

        write_csv(records.as_ref(), &shared.args.out).expect("write succeeded");

        let done = 1 + shared.done.fetch_add(1, Ordering::Relaxed);
        let total = shared.args.matches;

        let freq = (done as f64) / shared.start.elapsed().as_secs_f64();
        let secs = 1. / freq;

        let remaining = total - done;
        let eta = Duration::from_secs_f64(secs * (remaining as f64));

        let rate_fmt = if freq >= 1. {
            format!("{freq:.1} samples/sec")
        } else {
            format!("{secs:.1} secs/sample")
        };

        eprintln!("[{done:>5}/{total}] Rate: {rate_fmt} ETA: {eta:.1?}");
    }
}

fn write_csv(records: &[Record], path: &Path) -> io::Result<()> {
    let mut csv = File::create_buffered(path)?;

    writeln!(csv, "wins,draws,losses,white,black")?;

    for ((white, black), Record { wins, draws, losses }) in
        PAIRS.into_iter().zip(records)
    {
        writeln!(csv, "{wins},{draws},{losses},{white},{black}")?;
    }

    csv.flush()
}

fn build_game(
    seed: u64,
    white_team: &[CharacterBuilder],
    black_team: &[CharacterBuilder],
) -> Game {
    let map = Position::spiral(5).map(|p| (p, Tile::Open));
    let mut world = World::new(map, seed);

    for c in white_team {
        c.clone().team(Color::White).build(&mut world);
    }

    for c in black_team {
        c.clone().team(Color::Black).build(&mut world);
    }

    let rules = Rules::default();
    rules.on_start(&mut Start, &rules, &mut world);

    Game::new(world, rules, Some(1024))
}

fn play_game(
    mut game: Game,
    rng: &mut impl Rng,
    time: Option<Duration>,
    iters: Option<u32>,
    nodes: Option<usize>,
) -> Outcome {
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    while !game.is_over() {
        let mut moves = game.moves();
        let mov = if moves.len() == 1 {
            moves.next().unwrap()
        } else {
            let best = mcts.search(0, &game, rng, time, iters, nodes).unwrap();
            mcts.moves.keys()[mcts.nodes[best].mov]
        };

        drop(moves);

        game.play(mov);
        mcts = mcts.reroot_to_move(mov).unwrap_or_else(|| Mcts::new(mov));
    }

    game.result().unwrap()
}
