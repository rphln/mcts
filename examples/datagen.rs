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
    #[clap(default_value = "dist/")]
    pub destination: PathBuf,
    #[clap(long, default_value_t = 65_536)]
    pub iters: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    create_dir_all(&args.destination)?;

    let mut thread_rng = rand::rng();

    loop {
        let seed = thread_rng.next_u64();
        let mut game = Game::new(seed, true)?;

        let mut search_rng = DefaultRng::seed_from_u64(seed);
        let mut samples = vec![];

        let start_time = Instant::now();

        while !game.is_over() {
            let mov = search(&mut game, &mut search_rng, None, Some(args.iters), None);
            game.play(mov);

            let mut white_health = vec![];
            let mut black_health = vec![];

            let mut white_ready_at = vec![];
            let mut black_ready_at = vec![];

            let active_team = game.world.active_team();

            for character in &game.world.characters {
                match character.team {
                    Color::White => {
                        white_health.push(character.current_health());
                        white_ready_at.push(character.ready_at);
                    }
                    Color::Black => {
                        black_health.push(character.current_health());
                        black_ready_at.push(character.ready_at);
                    }
                }
            }

            let sample = json!({
                "tick": game.world.tick,
                "team": active_team,
                "white_health": white_health,
                "black_health": black_health,
                "white_ready_at": white_ready_at,
                "black_ready_at": black_ready_at,
            });
            samples.push(sample);
        }

        assert!(game.is_over());

        let elapsed = start_time.elapsed();
        let outcome = match game.world.active_team() {
            Color::White => Outcome::WhiteWins,
            Color::Black => Outcome::BlackWins,
        };

        let value = json!({
            "outcome": outcome,
            "samples": samples,
            "meta": {
                "seed": seed,
                "turns": samples.len(),
                "elapsed": elapsed.as_millis(),
                "search_iters": args.iters,
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
    let &mov = mcts.moves.resolve(mov).unwrap();

    mov
}
