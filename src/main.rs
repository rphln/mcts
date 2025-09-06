#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![feature(int_roundings)]

pub mod game;
pub mod mcts;

use std::{sync::mpsc, thread, time::Instant};

use rand::{SeedableRng, seq::IndexedRandom};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use swift_swallow::{
    Rng,
    command::Command,
    examples::setup,
    team::TeamKey::{self, Black, White},
};

use crate::{game::Game, mcts::Mcts};

fn main() {
    // region: Configuration.

    let search_baseline = search_mcts;
    let search_candidate = search_mcts;

    let elo_0 = 0.;
    let elo_1 = 10.;

    let alpha = 0.05;
    let beta = 0.10;

    // endregion
    // region: Generate games.

    let results = into_seq_iter((0..10_000).into_par_iter().map(move |seed| {
        let mut w = 0;
        let mut d = 0;
        let mut l = 0;

        match run_match(search_candidate, search_baseline, seed) {
            Some(White) => w += 1,
            Some(Black) => l += 1,
            None => d += 1,
        }

        match run_match(search_baseline, search_candidate, seed) {
            Some(Black) => w += 1,
            Some(White) => l += 1,
            None => d += 1,
        }

        (w, d, l)
    }));

    // endregion
    // region: SPRT.

    let now = Instant::now();

    let p_0 = elo(elo_0); // H₀: Candidate <= Baseline
    let p_1 = elo(elo_1); // H₁: Candidate > Baseline

    let lower = f64::ln(beta / (1. - alpha));
    let upper = f64::ln((1. - beta) / alpha);

    let mut llr = 0.;

    // region: Tally.

    let mut w = 0;
    let mut d = 0;
    let mut l = 0;

    let mut ww = 0;
    let mut wd = 0;
    let mut dd = 0;
    let mut ld = 0;
    let mut ll = 0;

    // endregion

    for (r_w, r_d, r_l) in results {
        let elapsed = now.elapsed();

        w += r_w;
        d += r_d;
        l += r_l;

        match (r_w, r_d, r_l) {
            (2, 0, 0) => ww += 1,
            (1, 1, 0) => wd += 1,
            (0, 2, 0) | (1, 0, 1) => dd += 1,
            (0, 1, 1) => ld += 1,
            (0, 0, 2) => ll += 1,
            (_, _, _) => unreachable!("{r_w}, {r_d}, {r_l}"),
        }

        let y = f64::from(r_w) / 2. + f64::from(r_d) / 4.;
        llr += y * f64::ln(p_1 / p_0) + (1. - y) * f64::ln((1. - p_1) / (1. - p_0));

        println!();
        println!("Status | {elapsed:.2?} elapsed...");
        println!("LLR    | {llr:.2} ({lower:.2}, {upper:.2}) [{elo_0:.2}, {elo_1:.2}]");
        println!("Games  | N: {n} W: {w} D: {d} L: {l}", n = w + d + l);
        println!("Penta  | WW: {ww} WD: {wd} DD: {dd} LD: {ld} LL: {ll}");

        if llr <= lower {
            println!("Result | Accept H₀");
            break;
        } else if llr >= upper {
            println!("Result | Accept H₁");
            break;
        }
    }

    // endregion
}

fn run_match(
    search_white: impl Fn(&Game, Command, u64) -> Command,
    search_black: impl Fn(&Game, Command, u64) -> Command,
    seed: u64,
) -> Option<TeamKey> {
    let mut game = Game::new(setup(seed).unwrap(), White);
    let mut mov = Command::None { team: TeamKey::Black };

    let mut moves = 0;

    while !game.is_over() && moves < 1024 {
        moves += 1;

        mov = if game.color == TeamKey::White {
            search_white(&game, mov, seed)
        } else {
            search_black(&game, mov, seed)
        };

        game.play(mov);
    }

    if moves >= 1024 {
        return None;
    }

    Some(game.world.active_team())
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

// endregion

// region: Helpers.

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
