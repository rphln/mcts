#![feature(file_buffered)]

use std::{
    fs::{File, create_dir_all},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Result;
use clap::Parser;
use rand::{RngCore, SeedableRng, prelude::*};
use serde_json::json;
use swift_swallow_rpc::Outcome;
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::{Mcts, Node},
};

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(default_value = "data/")]
    pub destination: PathBuf,
    #[clap(long, default_value_t = 0)]
    pub seed: u64,
    #[clap(long, value_parser = parse_millis, required_unless_present_any = ["iters", "nodes"])]
    pub time: Option<Duration>,
    #[clap(long, required_unless_present_any = ["time", "nodes"])]
    pub iters: Option<u32>,
    #[clap(long, required_unless_present_any = ["time", "iters"])]
    pub nodes: Option<usize>,
}

fn parse_millis(arg: &str) -> Result<Duration, std::num::ParseIntError> {
    let millis = arg.parse::<u64>()?;
    Ok(Duration::from_millis(millis))
}

fn main() -> Result<()> {
    let args = Args::parse();
    create_dir_all(&args.destination)?;

    let mut global_rng = DefaultRng::seed_from_u64(args.seed);

    loop {
        let seed = global_rng.next_u64();
        let mut game = Game::new(seed, true)?;

        let mut search_rng = DefaultRng::seed_from_u64(seed);

        let mut white_healths: Vec<Vec<i32>> = Vec::new();
        let mut black_healths: Vec<Vec<i32>> = Vec::new();

        let start_time = Instant::now();

        while !game.is_over() {
            let mov =
                search(&mut game, &mut search_rng, args.time, args.iters, args.nodes);
            game.play(&mov);

            let mut curr_white = Vec::new();
            let mut curr_black = Vec::new();

            for character in &game.world.characters {
                match character.team {
                    Color::White => curr_white.push(character.current_health()),
                    Color::Black => curr_black.push(character.current_health()),
                }
            }

            white_healths.push(curr_white);
            black_healths.push(curr_black);
        }

        let elapsed = start_time.elapsed();
        let outcome = match game.world.active_team() {
            Color::White => Outcome::WhiteWins,
            Color::Black => Outcome::BlackWins,
        };

        let value = json!({
            "match_outcome": outcome,
            "health_white": white_healths,
            "health_black": black_healths,
            "meta": {
                "seed": seed,
                "turns": white_healths.len(),
                "elapsed": elapsed.as_millis(),
                "mcts": {
                    "time": args.time.map(|d| d.as_millis()),
                    "iters": args.iters,
                    "nodes": args.nodes,
                }
            }
        });

        let name = format!("{seed:016x}.json");
        let path = args.destination.join(name);

        let mut writer = File::create_buffered(&path)?;
        serde_json::to_writer_pretty(&mut writer, &value)?;
    }
}

/// Performs a search from the current game state and returns the selected move.
///
/// # Panics
///
/// Panics if no legal moves are available in the current game state.
#[must_use]
pub fn search(
    game: &mut Game,
    rng: &mut impl Rng,
    time: Option<Duration>,
    iters: Option<u32>,
    nodes: Option<usize>,
) -> Move {
    let mut moves = game.moves();
    let first = moves.next().expect("`moves` should be non-empty");

    // For now, just skip the search for singular moves. Later on, we could ponder
    // here.
    if moves.next().is_none() {
        return first;
    }

    drop(moves);

    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let &Node { mov, .. } = mcts
        .search(0, game, rng, time, iters, nodes)
        .expect("`node` should have children");
    let &mov = mcts.moves.lookup(mov);

    mov
}
