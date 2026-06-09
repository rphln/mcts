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
        Game { world, rules, color: Color::White, depth: 0, max_depth }
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
        /// Decisive state value.
        const Z: f64 = 256.0;

        /// Each character provides at least one “Strike” worth of value.
        const S: f64 = 6.0;

        let mate = match self.result() {
            Some(Outcome::Draw) => return 0.0,
            Some(Outcome::Victory(other)) => {
                if color == other {
                    Z
                } else {
                    -Z
                }
            }
            None => 0.0,
        };

        let utility: f64 = self
            .world
            .characters
            .iter()
            .filter(|character| !character.is_defeated())
            .map(|character| {
                let health = f64::from(character.current_health() + character.block);
                let utility = f64::sqrt(S) + f64::sqrt(health);

                if color == character.team { utility } else { -utility }
            })
            .sum();

        mate + utility
    }
}
