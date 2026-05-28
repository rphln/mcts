use std::io::{self, BufWriter, Write};

use rand::prelude::*;
use swift_swallow::{rules::rules, setup};
use swift_swallow_rpc::{Handshake, Request};
use swift_swallow_tree_search::{
    DefaultRng,
    game::{Color, Game, Move},
    mcts::Mcts,
};

fn main() -> anyhow::Result<()> {
    let game_seed = 0;

    let shuffle = false;
    let max_ply = None;

    let mut rng = DefaultRng::seed_from_u64(42);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    let mut game = {
        let rules = rules();
        let world = setup(game_seed, shuffle, &rules);

        Game::new(world, rules, max_ply)
    };

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
                mcts = Mcts::new(Move::None { team: Color::Black });
                game = {
                    let rules = rules();
                    let world = setup(args.seed, args.shuffle, &rules);

                    Game::new(world, rules, args.max_ply)
                };

                let res = ();
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Play(args) => {
                game.play(args.mov);
                mcts = mcts
                    .reroot_to_move(args.mov)
                    .unwrap_or_else(|| Mcts::new(args.mov));

                let res = game.result();
                serde_json::to_writer(&mut tx, &res)?;
            }
            Request::Search(args) => {
                let best = mcts
                    .search(0, &game, &mut rng, args.time, args.iters, args.nodes)
                    .unwrap();

                let node = &mcts.nodes[best];
                let (&mov, _history) = mcts.moves.get_index(node.mov).unwrap();

                serde_json::to_writer(&mut tx, &mov)?;
            }
        }

        tx.write_all(b"\n")?;
        tx.flush()?;
    }

    Ok(())
}
