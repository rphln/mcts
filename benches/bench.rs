use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use swift_swallow_tree_search::{
    game::{Color, Game, Move},
    mcts::{Mcts, Node},
};

fn search(nodes: usize) -> Option<Node> {
    let game = Game::new(0, false).unwrap();

    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(Move::None { team: Color::Black });

    mcts.search(0, &game, &mut rng, None, None, Some(nodes)).cloned()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("search 250k nodes", |b| b.iter(|| search(black_box(250_000))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
