use std::hash::Hash;

/// A two-player game that the search can drive.
///
/// Implementors expose legal-move generation, move application, terminal
/// detection, and a heuristic evaluation. The search treats the state as
/// opaque, interning only the [`GameState::Move`] values it needs to track.
pub trait GameState: Clone {
    /// A single move.
    type Move: Eq + Hash;

    /// A side to move.
    type Color;

    /// Enumerates the legal moves in the current state.
    fn moves(&self) -> impl Iterator<Item = Self::Move>;

    /// Applies `mov`, advancing the state by one ply.
    fn play(&mut self, mov: &Self::Move);

    /// Returns whether the game has reached a terminal state.
    fn is_over(&self) -> bool;

    /// Returns the side to move.
    fn color(&self) -> Self::Color;

    /// Returns the number of plies played so far.
    fn depth(&self) -> usize;

    /// Heuristic score from `color`'s perspective.
    fn evaluate(&self, color: Self::Color) -> f64;
}
