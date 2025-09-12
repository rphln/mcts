#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::io::{self, BufWriter, Write};

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
                let res = search(&game, args.nodes, args.seed);
                serde_json::to_writer(&mut tx, &res)?;
            } // endregion
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}

fn search(game: &Game, max_nodes: usize, seed: u64) -> Move {
    let legal_moves = game.moves();
    assert!(!legal_moves.is_empty(), "No legal moves.");

    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mut rng = Rng::seed_from_u64(seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let mut nodes = 0;

    while nodes < max_nodes {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        nodes += mcts.depth(node);

        // See <https://www.sciencedirect.com/science/article/pii/S0304397516302717>.
        for _ in 0..2 {
            let &mov =
                game.moves().choose(&mut rng).expect("`moves` should be non-empty");
            game.play(mov);

            nodes += 1;
        }

        let reward = game.evaluate();
        mcts.backward(node, reward);
    }

    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}
