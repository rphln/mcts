#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod game;
pub mod mcts;

/// Pins a specific generator for portability and reproducibility.
pub type Rng = rand_xoshiro::Xoshiro256PlusPlus;
