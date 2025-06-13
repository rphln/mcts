use std::{
    cmp::{Reverse, max},
    collections::VecDeque,
    mem::swap,
};

use rustc_hash::FxHashMap;

use crate::game::{Command, Game};

pub const INF: i32 = 3072;

pub fn search(
    depth: i32,
    pv: &mut VecDeque<Command>,
    history: &mut FxHashMap<Command, i32>,
    game: &Game,
) -> i32 {
    pvs(-INF, INF, depth, pv, history, game)
}

fn pvs(
    mut alpha: i32,
    beta: i32,
    mut depth: i32,
    pv: &mut VecDeque<Command>,
    history: &mut FxHashMap<Command, i32>,
    game: &Game,
) -> i32 {
    if depth <= 0 || game.is_over() {
        return game.evaluate_i32();
    }

    let mut best_score = -INF;
    let mut moves = game.moves();

    // See <https://www.chessprogramming.org/One_Reply_Extensions>.
    if moves.len() == 1 {
        depth += 1;
    }

    // See <https://www.chessprogramming.org/Internal_Iterative_Deepening>.
    if pv.is_empty() && depth > 3 {
        let _ = pvs(alpha, beta, depth - 3, pv, history, game);
    }

    let pv_move = pv.pop_front();
    if let Some(mov) = pv_move {
        let mut game = game.clone();
        game.play(mov);

        let score = -pvs(-beta, -alpha, depth - 1, pv, history, &game);
        pv.push_front(mov);

        best_score = max(score, best_score);
        alpha = max(score, alpha);

        let theta = history.entry(mov).or_default();
        let bonus = 300 * depth - 250;

        if alpha >= beta {
            *theta = ridge_descent(*theta, bonus);
            return best_score;
        }
        *theta = ridge_descent(*theta, -bonus);
    }

    moves.sort_unstable_by_key(|mov| Reverse(history.get(mov)));

    // Scratch buffer for the principal variation.
    let buf = &mut VecDeque::new();

    for mov in moves {
        // Revisiting the PV move is a 10× slowdown. Whoops.
        if pv_move.is_some_and(|other| mov == other) {
            continue;
        }

        let mut game = game.clone();
        game.play(mov);

        let mut score;

        // See <https://www.chessprogramming.org/Late_Move_Reductions>.
        // let r = depth.div_ceil(3);
        let r = 1;
        score = -pvs(-alpha - 1, -alpha, depth - r, buf, history, &game);

        if score > alpha && r > 1 {
            score = -pvs(-alpha - 1, -alpha, depth - 1, buf, history, &game);
        }

        if score > alpha && beta - alpha > 1 {
            score = -pvs(-beta, -alpha, depth - 1, buf, history, &game);
        }

        if score > best_score {
            swap(pv, buf);
            pv.push_front(mov);
        }

        buf.clear();

        best_score = max(score, best_score);
        alpha = max(score, alpha);

        let theta = history.entry(mov).or_default();
        let bonus = 300 * depth - 250;

        if alpha >= beta {
            *theta = ridge_descent(*theta, bonus);
            break;
        }
        *theta = ridge_descent(*theta, -bonus);
    }

    best_score
}

/// Gradient descent update for the ridge regression $(x-y)^2 + x^2$ with step
/// size $\alpha = 300^{-1}$.
///
/// This seemingly works better than the history gravity formula.
fn ridge_descent(pred: i32, target: i32) -> i32 {
    let grad = 2 * (pred - target) + 2 * pred;
    pred - grad / 300
}
