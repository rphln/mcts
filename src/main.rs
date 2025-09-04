#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;
pub mod pvs;

use std::{collections::VecDeque, sync::mpsc, thread, time::Instant};

use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use swift_swallow::{
    Rng,
    event::Command,
    examples::setup,
    team::TeamKey::{self, White},
};

use crate::{game::Game, mcts::Mcts, pvs::search};

fn main() {
    const MS: u128 = 1_000_000;

    // region: Configuration.

    let engine_a = search_mcts;
    let engine_b = search_mcts;

    let p_0 = elo(0.); // H₀
    let p_1 = elo(10.); // H₁

    let ns_per_move = MS * 10;

    let alpha = 0.05;
    let beta = 0.05;

    // endregion
    // region: Generate games.

    let tasks = (0..65536).into_par_iter().flat_map(move |seed| {
        let is_white_win =
            run_match(seed, ns_per_move, engine_a, engine_b) == TeamKey::White;

        let is_black_win =
            run_match(seed, ns_per_move, engine_b, engine_a) == TeamKey::Black;

        [is_white_win, is_black_win]
    });

    // endregion
    // region: SPRT.

    let now = Instant::now();

    let log_alpha = f64::ln(beta / (1. - alpha));
    let log_beta = f64::ln((1. - beta) / alpha);

    let mut llr = 0.;

    for (n_iter, is_win) in into_seq_iter(tasks).enumerate() {
        if is_win {
            llr += f64::ln(p_1 / p_0);
        } else {
            llr += f64::ln((1. - p_1) / (1. - p_0));
        }

        let time = now.elapsed();
        println!("n_iter = {n_iter:>4} log_lr = {llr:>6.3} time = {time:.3?}");

        if llr <= log_alpha {
            println!("Accept H₀",);
            break;
        } else if llr >= log_beta {
            println!("Accept H₁",);
            break;
        }
    }

    // endregion
}

fn run_match(
    seed: u64,
    ns_per_move: u128,
    mut search_white: impl FnMut(&Game, Command, u128) -> Command,
    mut search_black: impl FnMut(&Game, Command, u128) -> Command,
) -> TeamKey {
    let mut game = Game::new(setup(seed).unwrap(), White);
    let mut mov = Command::None { team: TeamKey::Black };

    while !game.is_over() {
        mov = if game.color == TeamKey::White {
            search_white(&game, mov, ns_per_move)
        } else {
            search_black(&game, mov, ns_per_move)
        };
        game.play(mov);
    }
    game.world.active_team()
}

fn elo(rating_diff: f64) -> f64 {
    1. / (1. + f64::powf(10., -rating_diff / 400.))
}

// region: Search strategies

fn search_pvs(game: &Game, _previous_move: Command, ns_per_move: u128) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let start_time = thread_time_ns();

    let mut pv = VecDeque::default();
    let mut history = FxHashMap::default();

    for depth in 1.. {
        let _score = search(depth, &mut pv, &mut history, game);
        if thread_time_ns() - start_time >= ns_per_move {
            break;
        }
    }

    pv.pop_front().expect("`best_move` should exist")
}

fn search_mcts(game: &Game, _previous_move: Command, ns_per_move: u128) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let start_time = thread_time_ns();

    let mut mcts = Mcts::new(Command::None { team: TeamKey::Black });
    let mut rng = Rng::from_os_rng();

    for it in 1.. {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);

        if it % 1_000 == 0 && thread_time_ns() - start_time >= ns_per_move {
            break;
        }
    }

    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}

fn make_mcts_with_memory() -> impl FnMut(&Game, Command, u128) -> Command {
    let mut mcts = Mcts::new(Command::None { team: TeamKey::Black });
    let mut rng = Rng::from_os_rng();

    move |game: &Game, previous_move: Command, ns_per_move: u128| {
        let start_time = thread_time_ns();

        if let Some(prev_node) =
            mcts.children().iter().find(|node| node.mov == previous_move)
        {
            mcts = mcts.clone_subtree(prev_node);
        } else {
            mcts = Mcts::new(previous_move);
        }

        for it in 1.. {
            let mut game = game.clone();
            let node = mcts.select_and_expand(&mut game, &mut rng);

            let reward = game.evaluate_f64();
            mcts.backward(node, reward);

            if it % 1_000 == 0 && thread_time_ns() - start_time >= ns_per_move {
                break;
            }
        }

        let best_child = mcts.best_child().expect("`best_child` should exist");
        let best_mov = best_child.mov;
        mcts = mcts.clone_subtree(best_child);
        best_mov
    }
}

// endregion

// region: Helpers.

/// Returns the current CPU time in nanoseconds.
///
/// # Panics
///
/// Panics if the system call to get the CPU time fails.
fn thread_time_ns() -> u128 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };

    unsafe {
        assert_ne!(
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut time),
            -1,
            "Failed to get CPU time"
        );
    };

    let tv_sec = u128::try_from(time.tv_sec).unwrap();
    let tv_nsec = u128::try_from(time.tv_nsec).unwrap();

    tv_sec * 1_000_000_000 + tv_nsec
}

fn into_seq_iter<P: ParallelIterator + 'static>(
    par_iter: P,
) -> impl Iterator<Item = P::Item> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(|| {
        let _ = par_iter.try_for_each_with(tx, |tx, value| tx.send(value));
    });

    rx.into_iter()
}

// endregion
