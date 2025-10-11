use std::{iter::once, rc::Rc};

use either::Either;
use swift_swallow::{
    content::{Rules, rules, setup},
    event::{Command, decide, query_actions},
    team::TeamKey,
    world::World,
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
    pub rules: Rc<Rules>,
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
    #[must_use]
    pub fn new(game_seed: u64, randomize_ready_at: bool) -> anyhow::Result<Self> {
        let rules = rules();
        let world = setup(game_seed, randomize_ready_at, &rules)?;

        Ok(Self { world, rules: Rc::new(rules), color: TeamKey::White, depth: 0 })
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
            let rules = self.rules.as_ref();
            let query = query_actions(rules, &self.world).unwrap();

            let iter = query.into_iter();
            Either::Left(iter)
        } else {
            let iter = once(Command::None { team: self.color });
            Either::Right(iter)
        }
    }

    /// Applies a move to the world, alternates `color` and increments `depth`.
    pub fn play(&mut self, &mov: &Move) {
        let rules = self.rules.as_ref();
        let _decide = decide(mov, rules, &mut self.world).unwrap();

        self.color = !self.color;
        self.depth += 1;
    }

    /// Heuristic evaluation from the perspective of `color`.
    #[must_use]
    pub fn evaluate(&self) -> f64 {
        let mut max = 0.;
        let mut min = 0.;

        for character in &self.world.characters {
            if character.is_defeated() {
                continue;
            }

            let health = f64::from(character.current_health());
            let score = f64::sqrt(64. * health);

            if character.team == self.color {
                max += score;
            } else {
                min += score;
            }
        }

        max - min
    }
}
