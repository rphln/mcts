#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;
pub mod pvs;
pub mod sarsa;

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

use crate::{game::Game, mcts::Mcts, pvs::search, sarsa::Sarsa};

const MS: u128 = 1_000_000;
const NS_PER_MOVE: u128 = 10 * MS;

fn main() {
    // region: Configuration.

    let search_baseline = search_mcts;
    let search_candidate = search_mcts;

    let p_0 = elo(0.); // H₀: Candidate = Baseline
    let p_1 = elo(10.); // H₁: Candidate > Baseline

    let alpha = 0.05;
    let beta = 0.05;

    // endregion
    // region: Generate games.

    let results = into_seq_iter((0..10_000).into_par_iter().map(move |seed| {
        let is_white_win =
            run_match(search_candidate, search_baseline, seed) == TeamKey::White;

        let is_black_win =
            run_match(search_baseline, search_candidate, seed) == TeamKey::Black;

        [is_white_win, is_black_win]
    }));

    // endregion
    // region: SPRT.

    let now = Instant::now();

    let log_alpha = f64::ln(beta / (1. - alpha));
    let log_beta = f64::ln((1. - beta) / alpha);

    let mut llr = 0.;

    for (it, results) in results.enumerate() {
        let num_wins: usize = results.iter().map(|&r| r as usize).sum();
        let x: f64 = [0., 0.5, 1.][num_wins];

        llr += x * f64::ln(p_1 / p_0) + (1. - x) * f64::ln((1. - p_1) / (1. - p_0));

        let elapsed = now.elapsed();
        println!("it = {it:>4}  llr = {llr:>6.3}  elapsed = {elapsed:>6.2?}");

        if llr <= log_alpha {
            println!("Accept H₀");
            break;
        } else if llr >= log_beta {
            println!("Accept H₁");
            break;
        }
    }

    // endregion
}

fn run_match(
    search_white: impl Fn(&Game, Command, u64) -> Command,
    search_black: impl Fn(&Game, Command, u64) -> Command,
    seed: u64,
) -> TeamKey {
    let mut game = Game::new(setup(seed).unwrap(), White);
    let mut mov = Command::None { team: TeamKey::Black };

    while !game.is_over() {
        mov = if game.color == TeamKey::White {
            search_white(&game, mov, seed)
        } else {
            search_black(&game, mov, seed)
        };

        game.play(mov);
    }

    game.world.active_team()
}

fn elo(rating_diff: f64) -> f64 {
    1. / (1. + f64::powf(10., -rating_diff / 400.))
}

// region: Search strategies

fn search_mcts(game: &Game, previous_move: Command, search_seed: u64) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mut mcts = Mcts::new(previous_move);
    let mut rng = Rng::seed_from_u64(search_seed);

    for _ in 0..4096 {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);
    }

    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}

fn search_sarsa(game: &Game, previous_move: Command, search_seed: u64) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let mut mcts = Sarsa::new(previous_move);
    let mut rng = Rng::seed_from_u64(search_seed);

    for _ in 0..4096 {
        let mut game = game.clone();
        let node = mcts.select_and_expand(&mut game, &mut rng);

        let reward = game.evaluate_f64();
        mcts.backward(node, reward);
    }

    mcts.best_child().map(|node| node.mov).expect("`best_move` should exist")
}

#[must_use]
pub fn search_pvs(game: &Game, _previous_move: Command, _search_seed: u64) -> Command {
    let legal_moves = game.moves();
    if legal_moves.len() == 1 {
        return legal_moves[0];
    }

    let start_time = thread_time_ns();

    let mut pv = VecDeque::default();
    let mut history = FxHashMap::default();

    let mut best_move = legal_moves[0];

    for depth in 1.. {
        let _score = search(depth, &mut pv, &mut history, game);
        if thread_time_ns() - start_time >= NS_PER_MOVE {
            assert!(depth > 1);
            break;
        }

        // Prevent the engine from making a move after overtime.
        best_move = pv.front().copied().expect("`best_move` should exist");
    }

    best_move
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

/// Transforms `par_iter` into a sequential iterator that yields items as they
/// are generated, in an arbitrary order.
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
