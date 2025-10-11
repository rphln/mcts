#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::{hint::black_box, time::Duration};

use criterion::Criterion;
use criterion_macro::criterion;
use rand::{SeedableRng, rngs::StdRng};
use swift_swallow_tree_search::{
    game::{Color, Game, Move},
    mcts::Mcts,
};

#[criterion]
fn bench_mcts(c: &mut Criterion) {
    c.bench_function("search", |b| {
        b.iter_custom(|max_iters| {
            let game = Game::new(0, false).unwrap();
            let mut rng = StdRng::seed_from_u64(0);

            let mut mcts = Mcts::new(Move::None { team: Color::Black });

            let start = thread_time();

            black_box(mcts.search(
                0,
                black_box(&game),
                black_box(&mut rng),
                None,
                Some(max_iters as u32),
                None,
            ));

            thread_time() - start
        });
    });
}

/// Returns the CPU time used by the current thread.
///
/// # Panics
///
/// This uses the `clock_gettime` system call with the `CLOCK_THREAD_CPUTIME_ID`
/// clock. Panics if the system call to get the CPU time fails.
#[must_use]
fn thread_time() -> Duration {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };

    unsafe {
        assert_ne!(
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut time),
            -1,
            "Failed to get CPU time"
        );
    };

    let secs = u64::try_from(time.tv_sec).unwrap();
    let nanos = u32::try_from(time.tv_nsec).unwrap();

    Duration::new(secs, nanos)
}
