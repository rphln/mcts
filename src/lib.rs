pub mod game;
pub(crate) mod interner;
pub mod mcts;

/// Pins a specific generator for portability and reproducibility.
pub type DefaultRng = rand_xoshiro::Xoshiro256PlusPlus;
