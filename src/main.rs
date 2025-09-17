use std::io::{self, BufWriter, Write};

use rand::prelude::*;
use swift_swallow::examples::setup;
use swift_swallow_rpc::{Handshake, Outcome, Request, Search};
use swift_swallow_tree_search::{
    Rng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

fn main() -> anyhow::Result<()> {
    let game_seed = 0;
    let mut game = Game::new(setup(game_seed).unwrap(), Color::White);

    let mut tx = BufWriter::new(io::stdout());

    for line in io::stdin().lines() {
        match serde_json::from_str(&line?)? {
            Request::Handshake(_args) => {
                let res = Handshake {
                    name: env!("CARGO_PKG_NAME").to_owned(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                };
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::IsReady(_args) => {
                let res = true;
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Reset(args) => {
                game = Game::new(setup(args.seed).unwrap(), Color::White);

                let res = ();
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Play(args) => {
                game.play(args.mov);

                // TODO: Handle draws.
                let res = match game.world.active_team() {
                    _ if !game.is_over() => None,
                    Color::White => Some(Outcome::WhiteWins),
                    Color::Black => Some(Outcome::BlackWins),
                };

                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Search(args) => {
                let res = search(&game, &args);
                serde_json::to_writer(&mut tx, &res)?;
            }
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}

#[must_use]
pub fn search(game: &Game, args: &Search) -> Move {
    let legal_moves = game.moves();
    assert!(!legal_moves.is_empty(), "No legal moves.");

    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mut rng = Rng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let next = mcts.search(0, game, &mut rng, args.time, args.iters, args.nodes);
    mcts.tree[next].mov
}
