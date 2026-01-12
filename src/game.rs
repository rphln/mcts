use std::iter::once;

use either::Either;
use swift_swallow::{
    Command, Rules, TeamKey, World, decide, query_commands, rules, setup,
};

/// Alias for a game move (re-export of [`Command`]).
pub type Move = Command;

/// Alias for a side to move (re-export of [`TeamKey`]).
pub type Color = TeamKey;

/// Minimal game wrapper around `swift_swallow` providing a turn-alternating
/// interface, legal-move generation, plies played, and a simple evaluation
/// function.
#[derive(Clone)]
pub struct Game {
    /// Underlying world state.
    pub world: World,
    /// Rules for action generation and execution.
    pub rules: Rules,
    /// Local turn indicator used to alternate sides every move, regardless of
    /// what `world.active_team()` reports.
    pub color: TeamKey,
    /// Total number of plies played.
    ///
    /// Increments by 1 per `play`, regardless of whether it is a real or fake
    /// move.
    pub depth: usize,
}

impl Game {
    /// Constructs a new game using the default rules and content setup.
    ///
    /// # Errors
    ///
    /// Propagates any errors encountered during world setup. See [`setup`].
    pub fn new(game_seed: u64, randomize_ready_at: bool) -> anyhow::Result<Self> {
        let rules = rules();
        let world = setup(game_seed, randomize_ready_at, &rules)?;

        Ok(Self { world, rules, color: TeamKey::White, depth: 0 })
    }

    /// Whether the underlying world considers the game finished.
    #[must_use]
    pub fn is_over(&self) -> bool {
        self.world.is_game_over()
    }

    /// Iterator over legal moves for the current `color`.
    #[must_use]
    pub fn moves(&self) -> impl ExactSizeIterator<Item = Move> {
        if self.world.active_team() == self.color {
            let query = query_commands(&self.rules, &self.world);

            let iter = query.includes.into_iter();
            Either::Left(iter)
        } else {
            let iter = once(Command::None { team: self.color });
            Either::Right(iter)
        }
    }

    /// Applies a move to the world, alternates `color` and increments `depth`.
    pub fn play(&mut self, &mov: &Move) {
        let _decide = decide(mov, &self.rules, &mut self.world);

        self.color = !self.color;
        self.depth += 1;
    }

    /// Heuristic score from `color`'s perspective.
    ///
    /// Returns `predict()` for White and `-predict()` for Black (symmetric for
    /// negamax).
    #[must_use]
    pub fn evaluate(&self, color: Color) -> f64 {
        match color {
            Color::White => self.predict(),
            Color::Black => -self.predict(),
        }
    }

    /// Estimated probability White wins: `sigmoid(predict())`.
    ///
    /// See [`Self::predict`] for model and training details.
    #[must_use]
    pub fn predict_proba(&self) -> f64 {
        sigmoid(self.predict())
    }

    /// Raw logit for White (positive → White, negative → Black).
    ///
    /// Weighted sum of active characters' features plus bias. Model fitted
    /// with `swift_swallow_tree_search.examples.fit` and validated via a
    /// train/test split.
    #[must_use]
    pub fn predict(&self) -> f64 {
        const BASE_WEIGHT: f64 = 0.56;
        const HEALTH_WEIGHT: f64 = 0.16;
        const READY_WEIGHT: f64 = 0.;

        const BIAS: f64 = 0.;

        let score: f64 = self
            .world
            .characters
            .iter()
            .filter(|character| !character.is_defeated())
            .map(|character| {
                let health = f64::from(character.current_health());
                let ready_at = f64::from(character.ready_at - self.world.tick);

                let score = BASE_WEIGHT
                    + HEALTH_WEIGHT * health.sqrt()
                    + READY_WEIGHT * ready_at;

                match character.team {
                    Color::White => score,
                    Color::Black => -score,
                }
            })
            .sum();

        score + BIAS
    }
}

/// Logistic sigmoid: `1.0 / (1.0 + (-x).exp())`.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
