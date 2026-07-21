#![feature(test)]
extern crate test;

use std::hint::black_box;

use rand::prelude::*;
use swift_swallow_tree_search::{GameState, Mcts, MctsOptions};
use test::Bencher;

/// Number of columns.
const WIDTH: u8 = 7;
/// Bits per column: six playable rows plus the sentinel row.
const STRIDE: usize = 7;
/// Total number of playable cells.
const CELLS: u8 = 42;
/// The six playable cells of the leftmost column.
const COLUMN: u64 = 0x3F;

/// A bitboard Connect Four, used to exercise the search in isolation without
/// depending on a concrete game.
///
/// Each side's stones occupy a `u64` under a seven-bit-per-column layout: six
/// playable rows plus a sentinel row, kept empty, that stops four-in-a-row runs
/// from wrapping across column boundaries. The whole state is two words, so the
/// clone the search performs on every iteration is nearly free.
#[derive(Copy, Clone)]
struct ConnectFour {
    /// Stones played so far; its parity gives the side to move. White moves
    /// first.
    plies: u8,
    /// White's stones.
    white: u64,
    /// Black's stones.
    black: u64,
}

impl ConnectFour {
    const fn new() -> ConnectFour {
        ConnectFour { white: 0, black: 0, plies: 0 }
    }

    fn occupied(&self) -> u64 {
        self.white | self.black
    }
}

/// Returns whether `stones` contains four in a row in any direction.
fn won(stones: u64) -> bool {
    [1, STRIDE, STRIDE - 1, STRIDE + 1].into_iter().any(|shift| {
        let pair = stones & (stones >> shift);
        pair & (pair >> (2 * shift)) != 0
    })
}

impl GameState for ConnectFour {
    type Move = u8;
    type Color = bool;

    fn moves(&self) -> impl Iterator<Item = u8> {
        let occupied = self.occupied();
        (0..WIDTH).filter(move |&column| {
            let cells = COLUMN << (usize::from(column) * STRIDE);
            (occupied & cells) != cells
        })
    }

    fn play(&mut self, &column: &u8) {
        let shift = usize::from(column) * STRIDE;
        // Adding the column's bottom cell carries up to the lowest empty row.
        let cell = (self.occupied() + (1u64 << shift)) & (COLUMN << shift);

        if self.color() {
            self.black |= cell;
        } else {
            self.white |= cell;
        }

        self.plies += 1;
    }

    fn is_over(&self) -> bool {
        self.plies == CELLS || won(self.white) || won(self.black)
    }

    fn color(&self) -> bool {
        self.plies & 1 != 0
    }

    fn depth(&self) -> usize {
        usize::from(self.plies)
    }

    fn evaluate(&self, color: bool) -> f64 {
        let (white, black) =
            if color { (self.black, self.white) } else { (self.white, self.black) };

        if won(white) {
            return 1.0;
        } else if won(black) {
            return -1.0;
        } else if self.plies == CELLS {
            return 0.0;
        }

        // Centre-column control, the single strongest static feature.
        let center = COLUMN << (3 * STRIDE);

        let white = (white & center).count_ones();
        let black = (black & center).count_ones();

        f64::tanh((f64::from(white) - f64::from(black)) / 3.0)
    }
}

fn search(nodes: usize) -> Option<usize> {
    let game = ConnectFour::new();

    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(0, MctsOptions::default());

    mcts.search(0, &game, &mut rng, None, None, Some(nodes))
}

#[bench]
fn bench_search_250k_nodes(b: &mut Bencher) {
    b.iter(|| search(black_box(250_000)));
}
