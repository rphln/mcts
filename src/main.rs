use std::io::{self, BufWriter, Write};

use rand::prelude::*;
use swift_swallow_rpc::{Handshake, Outcome, Request, Search};
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::{Mcts, Node},
};

fn main() -> anyhow::Result<()> {
    let game_seed = 0;
    let randomize_ready_at = false;

    let mut game = Game::new(game_seed, randomize_ready_at)?;

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
                game = Game::new(args.seed, args.randomize_ready_at)?;

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
                let res = search(&mut game, &args);
                serde_json::to_writer(&mut tx, &res)?;
            }
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}

/// Performs a search from the current game state and returns the selected move.
///
/// # Panics
///
/// Panics if no legal moves are available in the current game state.
#[must_use]
pub fn search(game: &mut Game, args: &Search) -> Move {
    let mut moves = game.moves();
    let first = moves.next().expect("`moves` should be non-empty");

    // For now, just skip the search for singular moves. Later on, we could ponder
    // here.
    if moves.next().is_none() {
        return first;
    }

    drop(moves);

    let mut rng = DefaultRng::seed_from_u64(args.seed);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let &Node { mov, .. } = mcts
        .search(0, game, &mut rng, args.time, args.iters, args.nodes)
        .expect("`node` should have children");
    let &mov = mcts.moves.lookup(mov);

    mov
}
