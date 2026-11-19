#![forbid(unsafe_code)]
//! Bounded Scoped-Thread Work Stealing Pool (`CAS` Token Bucket Preventing Nested Oversubscription).

#[cfg(not(target_arch = "wasm32"))]
static ACTIVE_EXTRA_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(not(target_arch = "wasm32"))]
struct WorkerReservation {
    count: usize,
    counter: &'static std::sync::atomic::AtomicUsize,
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkerReservation {
    fn try_acquire(want_extra: usize, max_extra_pool: usize) -> Self {
        Self::try_acquire_from(&ACTIVE_EXTRA_WORKERS, want_extra, max_extra_pool)
    }

    fn try_acquire_from(
        counter: &'static std::sync::atomic::AtomicUsize,
        want_extra: usize,
        max_extra_pool: usize,
    ) -> Self {
        use std::sync::atomic::Ordering;
        if want_extra == 0 || max_extra_pool == 0 {
            return Self { count: 0, counter };
        }
        let mut current = counter.load(Ordering::Relaxed);
        loop {
            let avail = max_extra_pool.saturating_sub(current);
            let grant = want_extra.min(avail);
            if grant == 0 {
                return Self { count: 0, counter };
            }
            match counter.compare_exchange_weak(
                current,
                current + grant,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Self {
                        count: grant,
                        counter,
                    };
                }
                Err(actual) => current = actual,
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WorkerReservation {
    fn drop(&mut self) {
        if self.count > 0 {
            self.counter
                .fetch_sub(self.count, std::sync::atomic::Ordering::Release);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static UNDER_LLVM_COV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn running_under_llvm_cov() -> bool {
    *UNDER_LLVM_COV.get_or_init(|| std::env::var_os("LLVM_PROFILE_FILE").is_some())
}

/// Evaluates `f(idx)` for `idx in 0..len` in parallel using scoped threads, preserving index order,
/// capped to at most `max_extra_pool` background workers globally.
pub(crate) fn par_map_indexed_with_cap<R: Send>(
    len: usize,
    max_extra_pool: usize,
    f: impl Fn(usize) -> R + Send + Sync,
) -> Vec<R> {
    if len <= 1 || max_extra_pool == 0 {
        return (0..len).map(&f).collect();
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..len).map(f).collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let want_extra = len.saturating_sub(1).min(max_extra_pool);
        let reservation = WorkerReservation::try_acquire(want_extra, max_extra_pool);
        let extra_workers = reservation.count;
        if extra_workers == 0 {
            return (0..len).map(f).collect();
        }

        let cursor = AtomicUsize::new(0);
        let outputs: Vec<Mutex<Option<R>>> = (0..len).map(|_| Mutex::new(None)).collect();
        let f_ref = &f;
        let outputs_ref = &outputs;
        let cursor_ref = &cursor;
        let drain_work = || {
            while let Some(idx) = {
                let i = cursor_ref.fetch_add(1, Ordering::Relaxed);
                (i < len).then_some(i)
            } {
                let res = f_ref(idx);
                if let Ok(mut slot) = outputs_ref[idx].lock() {
                    *slot = Some(res);
                }
            }
        };
        std::thread::scope(|scope| {
            for _ in 0..extra_workers {
                scope.spawn(drain_work);
            }
            drain_work();
        });
        drop(reservation);
        outputs
            .into_iter()
            .filter_map(|m| m.into_inner().ok().flatten())
            .collect()
    }
}

/// Evaluates `f(idx)` for `idx in 0..len` in parallel using scoped threads, preserving index order.
pub(crate) fn par_map_indexed<R: Send>(len: usize, f: impl Fn(usize) -> R + Send + Sync) -> Vec<R> {
    if len <= 1 {
        return (0..len).map(&f).collect();
    }
    #[cfg(target_arch = "wasm32")]
    {
        par_map_indexed_with_cap(len, 0, f)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let max_extra_pool = if running_under_llvm_cov() {
            0
        } else {
            std::thread::available_parallelism()
                .map_or(4, usize::from)
                .max(1)
                .saturating_sub(1)
        };
        par_map_indexed_with_cap(len, max_extra_pool, f)
    }
}

/// Transforms `items` in parallel using scoped threads while preserving input order.
pub(crate) fn par_map_vec<T: Send, R: Send>(
    items: Vec<T>,
    f: impl Fn(usize, T) -> R + Send + Sync,
) -> Vec<R> {
    let len = items.len();
    if len <= 1 {
        return items
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect();
    }
    #[cfg(target_arch = "wasm32")]
    {
        items
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Mutex;

        let slots: Vec<Mutex<Option<T>>> = items.into_iter().map(|x| Mutex::new(Some(x))).collect();
        par_map_indexed(len, |idx| {
            let item = slots[idx].lock().ok().and_then(|mut g| g.take())?;
            Some(f(idx, item))
        })
        .into_iter()
        .flatten()
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_map_indexed_and_vec_empty_singleton_and_ordered() {
        assert!(par_map_indexed(0, |i| i).is_empty());
        assert_eq!(par_map_indexed(1, |i| i + 10), vec![10]);
        assert_eq!(
            par_map_indexed(32, |i| i * 3),
            (0..32).map(|i| i * 3).collect::<Vec<_>>()
        );
        assert!(par_map_vec(Vec::<usize>::new(), |_, x| x).is_empty());
        assert_eq!(par_map_vec(vec![7_usize], |i, x| i + x), vec![7]);
        let items: Vec<usize> = (0..32).map(|i| i + 5).collect();
        assert_eq!(
            par_map_vec(items, |i, x| i * 100 + x),
            (0..32).map(|i| i * 101 + 5).collect::<Vec<_>>()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_par_map_global_cas_token_bucket_never_oversubscribes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TEST_BUCKET: AtomicUsize = AtomicUsize::new(0);

        let max_extra = std::thread::available_parallelism()
            .map_or(4, usize::from)
            .saturating_sub(1);
        assert_eq!(WorkerReservation::try_acquire(0, max_extra).count, 0);
        assert_eq!(WorkerReservation::try_acquire(2, 0).count, 0);

        // Deterministic isolated CAS token accounting (0 -> grant -> cap -> 0) immune to concurrent test threads
        assert_eq!(TEST_BUCKET.load(Ordering::Acquire), 0);
        let r1 = WorkerReservation::try_acquire_from(&TEST_BUCKET, 2, 3);
        assert_eq!(r1.count, 2);
        assert_eq!(TEST_BUCKET.load(Ordering::Acquire), 2);
        let r2 = WorkerReservation::try_acquire_from(&TEST_BUCKET, 5, 3);
        assert_eq!(r2.count, 1);
        assert_eq!(TEST_BUCKET.load(Ordering::Acquire), 3);
        let r3 = WorkerReservation::try_acquire_from(&TEST_BUCKET, 1, 3);
        assert_eq!(r3.count, 0);
        drop(r3);
        drop(r2);
        assert_eq!(TEST_BUCKET.load(Ordering::Acquire), 2);
        drop(r1);
        assert_eq!(TEST_BUCKET.load(Ordering::Acquire), 0);

        std::thread::scope(|s| {
            for outer in 0..8_usize {
                s.spawn(move || {
                    let res = par_map_indexed_with_cap(16, max_extra, |mid| {
                        let inner = par_map_indexed_with_cap(8, max_extra, |k| {
                            assert!(ACTIVE_EXTRA_WORKERS.load(Ordering::Acquire) <= max_extra);
                            outer * 1000 + mid * 10 + k
                        });
                        assert_eq!(
                            inner,
                            (0..8)
                                .map(|k| outer * 1000 + mid * 10 + k)
                                .collect::<Vec<_>>()
                        );
                        mid * 2
                    });
                    assert_eq!(res, (0..16).map(|m| m * 2).collect::<Vec<_>>());
                });
            }
        });
    }
}
