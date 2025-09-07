#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

use std::io;

use rand::{SeedableRng, seq::IndexedRandom};
use swift_swallow::{
    Rng,
    command::Command,
    examples::setup,
    team::TeamKey::{self, White},
};

use crate::{game::Game, mcts::Mcts};

fn main() -> anyhow::Result<()> {
    let mut game_seed = 0;
    let mut search_seed = 0;

    let mut game = Game::new(setup(game_seed).unwrap(), White);
    let mut previous_move = Command::None { team: TeamKey::Black };

    for line in io::stdin().lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_ascii_whitespace().collect();

        match tokens.as_slice() {
            // regioin: Handshake.
            ["ugi"] => {
                println!("id name swift_swallow");
                println!("id author rphln");
                println!("ugiok");
            }
            ["isready"] => {
                println!("readyok");
            }
            ["quit"] => break,
            // endregion
            // region: Configuration.
            ["setoption", "name", "game_seed", "value", seed] => {
                game_seed = u64::from_str_radix(seed, 16)?;
            }
            ["setoption", "name", "search_seed", "value", seed] => {
                search_seed = u64::from_str_radix(seed, 16)?;
            }
            // endregion
            // region: Gaming.
            ["uginewgame"] => {}
            ["position", "startpos", suffix @ ..] => {
                game = Game::new(setup(game_seed).unwrap(), White);
                previous_move = Command::None { team: TeamKey::Black };

                if let ["moves", moves @ ..] = suffix {
                    for mov in moves {
                        let tokens: Vec<&str> = mov.split('/').collect();
                        let Ok(mov) = Command::try_from(tokens.as_slice()) else {
                            anyhow::bail!("Bad play command: {tokens:?}");
                        };

                        game.play(mov);
                        previous_move = mov;
                    }
                }
            }
            ["play", mov] => {
                let tokens: Vec<&str> = mov.split('/').collect();
                let Ok(mov) = Command::try_from(tokens.as_slice()) else {
                    anyhow::bail!("Bad play command: {tokens:?}");
                };

                game.play(mov);
            }
            ["go", ..] => {
                let mov = search(&game, previous_move, search_seed);
                game.play(mov);

                println!("bestmove {mov}");
            }
            // endregion
            // region: Queries.
            ["query", "result"] => {
                let side = match game.world.active_team() {
                    _team if !game.is_over() => "none",
                    TeamKey::White => "p1win",
                    TeamKey::Black => "p2win",
                };

                println!("response {side}");
            }
            // endregion
            tokens => {
                anyhow::bail!("Invalid command in this state: {tokens:?}");
            }
        }
    }

    Ok(())
}

fn search(game: &Game, previous_move: Command, search_seed: u64) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mut mcts = Mcts::new(previous_move);
    let mut rng = Rng::seed_from_u64(search_seed);

    let mut nodes = 0;

    while nodes < 10_000 {
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

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);
    }

    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}
