#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::{
    cmp::Reverse,
    fs::File,
    io::{self, BufWriter, Write},
};

use rand::{SeedableRng, seq::IndexedRandom};
use swift_swallow::examples::setup;
use swift_swallow_rpc::{Handshake, Outcome, Play, Request};

use crate::{
    game::{Color, Game, Move},
    mcts::Mcts,
};

/// Pins a specific generator for portability and reproducibility.
type Rng = rand_xoshiro::Xoshiro256PlusPlus;

fn main() -> anyhow::Result<()> {
    let game_seed = 0;
    let mut game = Game::new(setup(game_seed).unwrap(), Color::White);

    let mut tx = BufWriter::new(io::stdout());

    for line in io::stdin().lines() {
        match serde_json::from_str(&line?)? {
            // region: Control messages.
            Request::Handshake(_req) => {
                let res = Handshake {
                    name: env!("CARGO_PKG_NAME").to_owned(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                };
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::IsReady(_req) => {
                let res = true;
                serde_json::to_writer(&mut tx, &res)?;
            }
            // endregion
            // region: Game messages.
            Request::Reset(position) => {
                game = Game::new(setup(position.seed).unwrap(), Color::White);

                let res = ();
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Play(Play { mov }) => {
                game.play(mov);

                // TODO: Handle draws.
                let res = match game.world.active_team() {
                    _ if !game.is_over() => None,
                    Color::White => Some(Outcome::WhiteWins),
                    Color::Black => Some(Outcome::BlackWins),
                };

                serde_json::to_writer(&mut tx, &res)?;
            }
            // endregion
            // region: Search messages.
            Request::Search(args) => {
                let res = search(&game, args.nodes, args.time, args.seed);
                serde_json::to_writer(&mut tx, &res)?;
            }

            Request::SearchAndPlot(args) => {
                let mcts = search_mcts(&game, args.nodes, args.time, args.seed);

                let mut file = File::create("public_html/index.html")?;

                writeln!(file, r"<!doctype html>")?;
                writeln!(file, r#"<html lang="en">"#)?;
                writeln!(file, r"  <head>")?;
                writeln!(file, r#"    <meta charset="utf-8" />"#)?;
                writeln!(
                    file,
                    r#"    <meta name="viewport" content="width=device-width,initial-scale=1" />"#
                )?;
                writeln!(file, r"    <title>Flamegraph</title>")?;
                writeln!(file, r#"    <link rel="stylesheet" href="/icicle.css"/>"#)?;
                writeln!(
                    file,
                    r#"    <script type="text/javascript" src="/icicle.js"></script>"#
                )?;
                writeln!(file, r"  </head>")?;
                writeln!(file, r"  <body>")?;

                let min_visits = mcts.tree[0].visits.isqrt();

                let mut max = 0.;

                for node in &mcts.tree {
                    if node.visits < min_visits {
                        continue;
                    }

                    max = f64::max(max, f64::abs(node.value));
                }

                visit_and_write(0, &mcts, min_visits, &mut file, -max, max)?;

                writeln!(file, r"  </body>")?;
                writeln!(file, r"</html>")?;

                let res = ();
                serde_json::to_writer(&mut tx, &res)?;
            } // endregion
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}

fn search(
    game: &Game,
    max_nodes: Option<usize>,
    max_time: Option<u64>,
    seed: u64,
) -> Move {
    let legal_moves = game.moves();
    assert!(!legal_moves.is_empty(), "No legal moves.");

    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mcts = search_mcts(game, max_nodes, max_time, seed);
    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}

fn search_mcts(
    game: &Game,
    max_nodes: Option<usize>,
    max_time: Option<u64>,
    seed: u64,
) -> Mcts {
    let mut rng = Rng::seed_from_u64(seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let mut nodes = 0;
    let now = thread_time_ms();

    while max_nodes.is_none_or(|n| nodes < n)
        && max_time.is_none_or(|t| thread_time_ms() - now < t)
    {
        let mut game = game.clone();
        let node = mcts.expand_and_select(&mut game, &mut rng);

        // Ignore the root (which is not played) in the counting.
        let depth = mcts.ancestors(node).count() - 1;
        nodes += depth;

        // See <https://www.sciencedirect.com/science/article/pii/S0304397516302717>.
        for _ in 0..2 {
            if game.is_over() {
                break;
            }

            let &mov =
                game.moves().choose(&mut rng).expect("`moves` should be non-empty");
            game.play(mov);

            nodes += 1;
        }

        let reward = game.evaluate();
        mcts.backward(node, reward);
    }

    mcts
}

/// Returns the current CPU time in nanoseconds.
///
/// # Panics
///
/// Panics if the system call to get the CPU time fails.
fn thread_time_ms() -> u64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };

    unsafe {
        assert_ne!(
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut time),
            -1,
            "Failed to get CPU time"
        );
    };

    let tv_sec = u64::try_from(time.tv_sec).unwrap();
    let tv_nsec = u64::try_from(time.tv_nsec).unwrap();

    tv_sec * 1_000 + tv_nsec / 1_000_000
}

// region: Plotting.

fn visit_and_write(
    id: usize,
    mcts: &Mcts,
    min_visits: u32,
    writer: &mut impl Write,
    min: f64,
    max: f64,
) -> std::io::Result<()> {
    let node = &mcts.tree[id];
    if node.visits < min_visits {
        return Ok(());
    }

    let label = label(&node.mov);
    let width = if node.is_root() {
        100.
    } else {
        let parent = &mcts.tree[node.parent];
        100. * f64::from(node.visits) / f64::from(parent.visits)
    };

    let value = (node.value - min) / (max - min);

    writeln!(writer, r#"<div class="node" style="--width: {width:.3}%">"#)?;
    writeln!(
        writer,
        r#"  <div class="bar" style="--value: {value:.3}" title="{label}">"#
    )?;
    writeln!(writer, r#"    <span class="label">{label}</span>"#)?;
    writeln!(writer, r"  </div>")?;
    writeln!(writer, r#"  <div class="children">"#)?;

    let mut children: Vec<usize> = (node.head..node.last).collect();
    children.sort_by_key(|&idx| Reverse(mcts.tree[idx].visits));

    for child in children {
        visit_and_write(child, mcts, min_visits, writer, min, max)?;
    }

    writeln!(writer, r"  </div>")?;
    writeln!(writer, r"</div>")?;

    Ok(())
}

const CHARACTER: [&str; 6] = [
    "⚪ Vanguard",
    "⚫ Vanguard",
    "⚪ Scholar",
    "⚫ Scholar",
    "⚪ Marksman",
    "⚫ Marksman",
];
const ACTION: [&str; 24] = [
    "⚪ Onslaught",
    "⚫ Onslaught",
    "⚪ Unmend",
    "⚫ Unmend",
    "⚪ Grit",
    "⚫ Grit",
    "⚪ Fang and Claw",
    "⚫ Fang and Claw",
    "⚪ Ruin",
    "⚫ Ruin",
    "⚪ Adloquium",
    "⚫ Adloquium",
    "⚪ Deployment Tactics",
    "⚫ Deployment Tactics",
    "⚪ Emergency Tactics",
    "⚫ Emergency Tactics",
    "⚪ Bloodletter",
    "⚫ Bloodletter",
    "⚪ Sidewinder",
    "⚫ Sidewinder",
    "⚪ Lock and Load",
    "⚫ Lock and Load",
    "⚪ Iron Jaws",
    "⚫ Iron Jaws",
];

fn label(mov: &Move) -> String {
    match mov {
        Move::None { team } => match team {
            Color::White => "⚪".to_owned(),
            Color::Black => "⚫".to_owned(),
        },
        &Move::Pass { team, can_act, can_move } => {
            let color = match team {
                Color::White => "⚪",
                Color::Black => "⚫",
            };

            let marker = if can_act && can_move {
                "‼️"
            } else if can_act || can_move {
                "❗"
            } else {
                ""
            };

            format!("{color} Pass {marker}")
        }
        Move::Movement { character, destination } => {
            format!(
                "{label} → ⟨{x}, {y}⟩",
                label = CHARACTER[character.0],
                x = destination.x,
                y = destination.y,
            )
        }
        Move::Action { action, destination } => {
            format!(
                "{label} → ⟨{x}, {y}⟩",
                label = ACTION[action.0],
                x = destination.x,
                y = destination.y,
            )
        }
    }
}

// endregion
