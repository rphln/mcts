use std::{
    array,
    fmt::Display,
    panic::catch_unwind,
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicUsize, Ordering},
    },
    thread,
    time::Instant,
};

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

const NUM_WORKERS: usize = 16;

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
    // [0, 1, 2, 3] = Paladin, Warrior, Scholar, Sidewinder
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(4, 1)],
    // [0, 1, 2, 4] = Paladin, Warrior, Scholar, Rogue
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    // [0, 1, 2, 5] = Paladin, Warrior, Scholar, Warlock
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    // [0, 1, 3, 4] = Paladin, Warrior, Sidewinder, Rogue
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    // [0, 1, 3, 5] = Paladin, Warrior, Sidewinder, Warlock
    [qr(2, 0), qr(2, -1), qr(4, -1), qr(3, 0)],
    // [0, 1, 4, 5] = Paladin, Warrior, Rogue, Warlock
    [qr(2, 0), qr(2, -1), qr(3, 0), qr(3, -1)],
    // [0, 2, 3, 4] = Paladin, Scholar, Sidewinder, Rogue
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    // [0, 2, 3, 5] = Paladin, Scholar, Sidewinder, Warlock
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    // [0, 2, 4, 5] = Paladin, Scholar, Rogue, Warlock
    [qr(2, 0), qr(4, -1), qr(3, 0), qr(3, -1)],
    // [0, 3, 4, 5] = Paladin, Sidewinder, Rogue, Warlock
    [qr(2, 0), qr(4, 1), qr(3, 0), qr(3, -1)],
    // [1, 2, 3, 4] = Warrior, Scholar, Sidewinder, Rogue
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    // [1, 2, 3, 5] = Warrior, Scholar, Sidewinder, Warlock
    [qr(2, 0), qr(4, -1), qr(4, 1), qr(3, 0)],
    // [1, 2, 4, 5] = Warrior, Scholar, Rogue, Warlock
    [qr(2, 0), qr(4, -1), qr(3, 0), qr(3, -1)],
    // [1, 3, 4, 5] = Warrior, Sidewinder, Rogue, Warlock
    [qr(2, 0), qr(4, 1), qr(3, 0), qr(3, -1)],
    // [2, 3, 4, 5] = Scholar, Sidewinder, Rogue, Warlock
    [qr(4, -1), qr(4, 1), qr(3, 0), qr(3, -1)],
];

// Ordered to even out the character distributions at each given timestep.
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

#[derive(Default)]
struct State {
    next_match: AtomicUsize,
    curr_match: AtomicUsize,
    cross: [[Beta; 6]; 6],
    intra: [[Beta; 6]; 6],
}

fn main() {
    let mut rng = DefaultRng::seed_from_u64(42);

    let start = Instant::now();
    let state = Arc::new(State::default());

    for _ in 0..NUM_WORKERS {
        let mut rng = DefaultRng::from_rng(&mut rng);
        let state = state.clone();

        thread::spawn(move || {
            let characters = [
                characters::paladin(),
                characters::warrior(),
                characters::scholar(),
                characters::sidewinder(),
                characters::rogue(),
                characters::warlock(),
            ];

            loop {
                let n = state.next_match.fetch_add(1, Ordering::SeqCst);

                let (white_chr, black_chr, white_team, black_team) = {
                    let (white, black) = PAIRS[n % PAIRS.len()];

                    let white_chr = TEAMS[white];
                    let black_chr = TEAMS[black];

                    let white_pos = POSITIONS[white];
                    let black_pos = POSITIONS[black].map(|p| -p);

                    let white_team: [CharacterBuilder; 4] = array::from_fn(|idx| {
                        characters[white_chr[idx]].clone().position(white_pos[idx])
                    });

                    let black_team: [CharacterBuilder; 4] = array::from_fn(|idx| {
                        characters[black_chr[idx]].clone().position(black_pos[idx])
                    });

                    (white_chr, black_chr, white_team, black_team)
                };

                let seed = rng.next_u64();
                let outcome = catch_unwind(|| {
                    let game = build_game(seed, &white_team, &black_team);

                    let mut mcts_rng = DefaultRng::seed_from_u64(seed);
                    play_game(game, &mut mcts_rng)
                })
                .unwrap_or_else(|err| {
                    panic!("panic: seed={seed} n={n} err={err:?}");
                });

                let white_won = matches!(outcome, Outcome::Victory(Color::White));

                for w in white_chr {
                    for b in black_chr {
                        if white_won {
                            state.cross[w][b].win();
                            state.cross[b][w].loss();
                        } else {
                            state.cross[w][b].loss();
                            state.cross[b][w].win();
                        }
                    }
                }

                for i in white_chr {
                    for j in white_chr {
                        if white_won {
                            state.intra[i][j].win();
                        } else {
                            state.intra[i][j].loss();
                        }
                    }
                }

                for i in black_chr {
                    for j in black_chr {
                        if white_won {
                            state.intra[i][j].loss();
                        } else {
                            state.intra[i][j].win();
                        }
                    }
                }

                let n = state.curr_match.fetch_add(1, Ordering::SeqCst);
                let elapsed = start.elapsed();

                println!("Elapsed: {elapsed:.2?}");
                println!("Matches: {n}");
                println!();

                println!("Adversaries");
                print_matrix(&characters, &state.cross);
                println!();

                println!("Teammates");
                print_matrix(&characters, &state.intra);
                println!();
            }
        });
    }

    thread::park();
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

    let () = rules.on_start(&mut Start, &rules, &mut world);

    Game::new(world, rules, None)
}

fn play_game(mut game: Game, rng: &mut impl Rng) -> Outcome {
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    while !game.is_over() {
        let best = mcts
            .search_while(0, &game, rng, |mcts, _node, _game| {
                mcts.nodes[0].visits < 400_000
            })
            .expect("`root` should have children");

        let mov = mcts.moves.keys()[mcts.nodes[best].mov];
        game.play(mov);

        mcts.reroot(best);
    }

    game.result().unwrap()
}

fn print_cell(value: impl Display) {
    print!(" {value:>12}");
}

fn print_matrix(roster: &[CharacterBuilder], matrix: &[[Beta; 6]]) {
    print_cell("");

    for c in roster {
        print_cell(&c.name);
    }

    println!();

    for (src, row) in roster.iter().zip(matrix.iter()) {
        print_cell(&src.name);

        for beta in row {
            let (mean, std) = beta.mean_std();
            print_cell(format!("{mean:.2}±{std:.2}"));
        }

        println!();
    }
}

struct Beta {
    alpha: AtomicU32,
    beta: AtomicU32,
}

impl Default for Beta {
    fn default() -> Self {
        Self { alpha: AtomicU32::new(1), beta: AtomicU32::new(1) }
    }
}

impl Beta {
    fn win(&self) {
        self.alpha.fetch_add(1, Ordering::Relaxed);
    }

    fn loss(&self) {
        self.beta.fetch_add(1, Ordering::Relaxed);
    }

    fn mean_std(&self) -> (f64, f64) {
        let alpha = f64::from(self.alpha.load(Ordering::SeqCst));
        let beta = f64::from(self.beta.load(Ordering::SeqCst));

        let mean = alpha / (alpha + beta);

        let var = mean * (1.0 - mean) / (alpha + beta + 1.0);
        let std = var.sqrt();

        (mean, std)
    }
}
