#![feature(test)]
extern crate test;

use std::hint::black_box;

use rand::prelude::*;
use swift_swallow_core::{rules::rules, setup};
use swift_swallow_tree_search::{
    game::{Color, Game, Move},
    mcts::Mcts,
};
use test::Bencher;

fn search(nodes: usize) -> Option<usize> {
    let game = {
        let rules = rules();
        let world = setup(0, false, &rules);

        Game::new(world, rules, None)
    };

    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    mcts.search(0, &game, &mut rng, None, None, Some(nodes))
}

#[bench]
fn bench_search_250k_nodes(b: &mut Bencher) {
    b.iter(|| search(black_box(250_000)));
}
