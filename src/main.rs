#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::io::{self, BufWriter, Write};

use rand::{SeedableRng, seq::IndexedRandom};
use swift_swallow::{
    Rng,
    command::Command,
    examples::setup,
    team::TeamKey::{self, White},
};
use swift_swallow_rpc::{Handshake, Outcome, Play, Request};

use crate::{game::Game, mcts::Mcts};

fn main() -> anyhow::Result<()> {
    let game_seed = 0;
    let mut game = Game::new(setup(game_seed).unwrap(), White);

    let mut mcts = Mcts::new(Command::None { team: TeamKey::Black });

    let mut root_node = 0;
    let mut root_depth = 0;

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
                game = Game::new(setup(position.seed).unwrap(), White);

                mcts = Mcts::new(Command::None { team: TeamKey::Black });

                root_node = 0;
                root_depth = 0;

                let res = ();
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Play(Play { mov }) => {
                // region: Traverse the MCTS tree.

                let head = mcts.tree[root_node].head;
                let last = mcts.tree[root_node].last;

                if let Some(next) =
                    (head..last).into_iter().find(|&next| mcts.tree[next].mov == mov)
                {
                    root_node = next;
                    root_depth += 1;

                    assert!(root_node >= head);
                    assert!(root_node < last);
                } else {
                    mcts = Mcts::new(mov);

                    root_node = 0;
                    root_depth = 0;
                }

                // endregion

                game.play(mov);

                // TODO: Handle draws.
                let res = match game.world.active_team() {
                    _ if !game.is_over() => None,
                    TeamKey::White => Some(Outcome::WhiteWins),
                    TeamKey::Black => Some(Outcome::BlackWins),
                };

                serde_json::to_writer(&mut tx, &res)?;
            }
            // endregion
            // region: Search messages.
            Request::Search(args) => {
                let mut rng = Rng::seed_from_u64(args.seed);

                let legal_moves = game.moves();
                let res = if legal_moves.len() == 1 {
                    legal_moves[0]
                } else {
                    let mut nodes = 0;

                    while nodes < args.nodes {
                        let mut game = game.clone();
                        let node =
                            mcts.select_and_expand_node(root_node, &mut game, &mut rng);

                        nodes += mcts.depth(node) - root_depth;

                        // See <https://www.sciencedirect.com/science/article/pii/S0304397516302717>.
                        for _ in 0..2 {
                            let &mov = game
                                .moves()
                                .choose(&mut rng)
                                .expect("`moves` should be non-empty");
                            game.play(mov);

                            nodes += 1;
                        }

                        let reward = game.evaluate_f64();
                        mcts.backward(node, reward);
                    }

                    mcts.best_child()
                        .map(|node| node.mov)
                        .expect("`best_move` should exist")
                };

                serde_json::to_writer(&mut tx, &res)?;
            } // endregion
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}
