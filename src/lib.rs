pub mod game;
pub mod mcts;
pub mod normal_inverse_gamma;

/// Pins a specific generator for portability and reproducibility.
pub type DefaultRng = rand_xoshiro::Xoshiro256PlusPlus;
