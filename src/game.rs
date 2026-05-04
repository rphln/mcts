use std::iter::once;

use either::Either;
use swift_swallow::{
    Command, Outcome, Rules, TeamKey, World, decide, query_commands, rules, setup,
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
    pub color: Color,
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

        Ok(Self { world, rules, color: Color::White, depth: 0 })
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
    pub fn play(&mut self, mov: Move) -> Option<Outcome> {
        let _decide = decide(mov, &self.rules, &mut self.world);

        self.color = !self.color;
        self.depth += 1;

        self.world.outcome()
    }

    /// Heuristic score from `color`'s perspective.
    #[must_use]
    pub fn evaluate(&self, color: Color) -> f64 {
        const W0: f64 = -0.216;
        const W1: f64 = 0.454;

        if let Some(Outcome::Draw) = self.world.outcome() {
            return 0.;
        }

        self.world
            .characters
            .iter()
            .filter(|character| !character.is_defeated())
            .map(|character| {
                let health = f64::from(character.current_health());

                let score = W0 + W1 * f64::sqrt(health);
                if character.team == color { score } else { -score }
            })
            .sum()
    }
}
