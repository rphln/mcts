#![feature(assert_matches)]
#![feature(explicit_tail_calls)]

pub mod game;
pub mod mcts;

/// Pins a specific generator for portability and reproducibility.
pub type DefaultRng = rand_xoshiro::Xoshiro256PlusPlus;
