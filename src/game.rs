use std::cmp::max;

use swift_swallow::{Command, Outcome, Rules, TeamKey, World, decide, query_commands};

/// Alias for a game move (re-export of [`Command`]).
pub type Move = Command;

/// Alias for a side to move (re-export of [`TeamKey`]).
pub type Color = TeamKey;

/// A game wrapper around `swift_swallow` that enforces strict turn alternation,
/// generates legal moves, and tracks plies.
///
/// Each `play` call increments the ply count, even if the move is a no-op due
/// to turn enforcement.
#[derive(Clone)]
pub struct Game {
    pub world: World,
    pub rules: Rules,
    pub color: Color,
    pub depth: usize,
    pub max_depth: Option<usize>,
}

impl Game {
    #[must_use]
    pub fn new(world: World, rules: Rules, max_depth: Option<usize>) -> Game {
        Self { world, rules, color: Color::White, depth: 0, max_depth }
    }

    #[must_use]
    pub fn moves(&self) -> impl ExactSizeIterator<Item = Move> {
        let moves = if self.world.active_team() == self.color {
            let query = query_commands(&self.rules, &self.world);
            query.includes
        } else {
            vec![Command::None { team: self.color }]
        };

        moves.into_iter()
    }

    pub fn play(&mut self, mov: Move) {
        self.color = !self.color;
        self.depth += 1;

        if let Move::None { .. } = mov {
            return;
        }

        decide(mov, &self.rules, &mut self.world);
    }

    #[must_use]
    pub fn result(&self) -> Option<Outcome> {
        if self.max_depth.is_some_and(|max| self.depth >= max) {
            Some(Outcome::Draw)
        } else {
            self.world.result()
        }
    }

    #[must_use]
    pub fn is_over(&self) -> bool {
        self.result().is_some()
    }

    /// Heuristic score from `color`'s perspective.
    #[must_use]
    pub fn evaluate(&self, color: Color) -> f64 {
        /// Actions must provide at least one “Strike” worth of value as the
        /// baseline.
        const P: f64 = 6.0;

        /// Future action discount factor; i.e, per-tick “likelihood” of an
        /// action becoming available again. Uncalibrated.
        const S: f64 = 0.8;

        /// Reward for guaranteed wins.
        const MATE: f64 = 512.0;

        let mut friends = 0.0;
        let mut enemies = 0.0;

        let mate = match self.result() {
            None => 0.0,
            Some(Outcome::Victory(other)) => {
                if other == color {
                    MATE
                } else {
                    -MATE
                }
            }
            Some(Outcome::Draw) => {
                return 0.0;
            }
        };

        for character in &self.world.characters {
            if character.is_defeated() {
                continue;
            }

            let utility = f64::from(character.current_health() + character.block);

            if character.team == color {
                friends += utility;
            } else {
                enemies += utility;
            }
        }

        for action in &self.world.actions {
            let character = &self.world.characters[action.character.0];
            if character.is_defeated() {
                continue;
            }

            let ready_in = max(0, action.ready_at - self.world.tick);
            let utility = P * f64::powi(S, ready_in);

            if character.team == color {
                friends += utility;
            } else {
                enemies += utility;
            }
        }

        mate + friends - enemies
    }
}
