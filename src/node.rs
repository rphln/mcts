/// Placeholder for an unset node index.
pub const SENTINEL: usize = !0;

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct Node {
    /// Move used to reach this node.
    pub mov: usize,
    /// Parent index.
    pub parent: usize,
    /// First child index (inclusive).
    pub head: usize,
    /// Last child index (exclusive).
    pub last: usize,
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
}

impl Node {
    /// Creates a new node with default statistics.
    #[must_use]
    pub(crate) const fn new(parent: usize, mov: usize) -> Node {
        Node { mov, parent, head: SENTINEL, last: SENTINEL, visits: 0, value: 0.0 }
    }

    /// Returns whether this node is the root node.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent == SENTINEL
    }

    /// Returns whether this node is a leaf.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.head == self.last
    }
}
