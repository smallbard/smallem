//! Un groupe de fils permanent, et le partage du travail entre eux.
//!
//! Créer un fil coûte des dizaines de microsecondes, autant qu'une petite
//! opération de nos modèles. Les fils sont donc créés une fois, et attendent
//! du travail. Un appel à [`Pool::run`] leur confie `tasks` tâches numérotées,
//! que chacun prend tour à tour dans un compteur commun, et attend qu'elles
//! soient toutes faites. Le fil appelant travaille avec eux.
//!
//! Le découpage en tâches est fixé par l'appelant, d'après la forme du
//! tenseur, jamais d'après le nombre de fils : quelle que soit la façon dont
//! les tâches se répartissent, chacune fait le même calcul, et le résultat
//! est le même au bit près.

use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Combien de tours un fil inoccupé attend activement avant de s'endormir :
/// de l'ordre de 50 µs. Entre deux opérations d'un pas, le travail revient
/// bien plus vite, et un fil qui tourne repart en une fraction de
/// microseconde, là où un fil endormi demande un appel au système.
const SPINS: usize = 1 << 14;

thread_local! {
    /// Vrai dans une tâche : un appel imbriqué s'exécute alors sur place,
    /// sans redistribuer le travail.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// Une tâche, vue par les fils : la fermeture de l'appel en cours, dont la
/// durée de vie est effacée. [`Pool::run`] ne rend la main qu'une fois tous
/// les fils sortis de la tâche : la fermeture vit plus longtemps qu'eux.
#[derive(Clone, Copy)]
struct Job {
    work: *const (dyn Fn(usize) + Sync),
    tasks: usize,
}

// Le pointeur ne sert que pendant `Pool::run`, qui le garde vivant.
unsafe impl Send for Job {}

struct Shared {
    job: Mutex<Option<Job>>,
    /// Incrémenté à chaque appel : c'est ainsi qu'un fil voit du travail.
    epoch: AtomicUsize,
    /// La prochaine tâche à prendre.
    next: AtomicUsize,
    /// Les fils qui n'ont pas encore fini l'appel en cours.
    pending: AtomicUsize,
    sleepers: AtomicUsize,
    wake: Condvar,
    sleep: Mutex<()>,
    stop: AtomicBool,
    panicked: AtomicBool,
}

impl Shared {
    /// Prend et exécute des tâches jusqu'à ce qu'il n'en reste plus.
    fn work(&self, job: Job) {
        // SAFETY : `Pool::run` attend que tous les fils aient fini.
        let work = unsafe { &*job.work };
        INSIDE.with(|inside| inside.set(true));
        loop {
            let task = self.next.fetch_add(1, Ordering::Relaxed);
            if task >= job.tasks {
                break;
            }
            if catch_unwind(AssertUnwindSafe(|| work(task))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
        }
        INSIDE.with(|inside| inside.set(false));
    }
}

/// Le groupe de fils.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Un seul appel à la fois.
    calls: Mutex<()>,
}

impl Pool {
    /// Un groupe de `threads` fils en tout, le fil appelant compris.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            epoch: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            wake: Condvar::new(),
            sleep: Mutex::new(()),
            stop: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
        });
        let workers = (1..threads.max(1))
            .map(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("smallem-{index}"))
                    .spawn(move || worker(&shared))
                    .expect("impossible de créer un fil")
            })
            .collect();
        Self {
            shared,
            workers,
            calls: Mutex::new(()),
        }
    }

    /// Nombre de fils, le fil appelant compris.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Exécute `work(0)`, …, `work(tasks - 1)`, réparties entre les fils, et
    /// attend qu'elles soient toutes faites.
    pub fn run(&self, tasks: usize, work: impl Fn(usize) + Sync) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.workers.is_empty() || INSIDE.with(Cell::get) {
            (0..tasks).for_each(work);
            return;
        }
        let _call = self.calls.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &*self.shared;
        let work: &(dyn Fn(usize) + Sync) = &work;
        // SAFETY : on efface la durée de vie de la fermeture ; on attend
        // plus bas que plus aucun fil ne s'en serve.
        let work: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(work) };
        let job = Job { work, tasks };
        *shared.job.lock().unwrap() = Some(job);
        shared.next.store(0, Ordering::Relaxed);
        shared.pending.store(self.workers.len(), Ordering::SeqCst);
        shared.epoch.fetch_add(1, Ordering::SeqCst);
        if shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = shared.sleep.lock().unwrap();
            shared.wake.notify_all();
        }
        shared.work(job);
        let mut spins = 0;
        while shared.pending.load(Ordering::Acquire) > 0 {
            if spins < SPINS {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        if shared.panicked.swap(false, Ordering::Relaxed) {
            resume_unwind(Box::new("une tâche du groupe de fils a paniqué"));
        }
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        // Attendre un nouvel appel : d'abord activement, puis endormi.
        let mut spins = 0;
        loop {
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            if shared.epoch.load(Ordering::SeqCst) != seen {
                break;
            }
            if spins < SPINS {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let mut guard = shared.sleep.lock().unwrap();
            shared.sleepers.fetch_add(1, Ordering::SeqCst);
            while shared.epoch.load(Ordering::SeqCst) == seen && !shared.stop.load(Ordering::SeqCst)
            {
                guard = shared.wake.wait(guard).unwrap();
            }
            shared.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        seen = shared.epoch.load(Ordering::SeqCst);
        let job = shared.job.lock().unwrap().expect("un appel sans tâche");
        shared.work(job);
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _guard = self.shared.sleep.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Un tampon qu'on peut partager entre fils, pour que chaque tâche écrive
/// sa propre partie : un pointeur, la longueur du tampon, et l'emprunt dont
/// il vient, qui l'empêche de survivre au tampon.
///
/// Les bornes sont vérifiées ; seul le partage reste à la charge de
/// l'appelant : deux tâches ne doivent jamais écrire au même endroit.
pub struct SharedMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _borrow: PhantomData<&'a mut [T]>,
}

impl<T> Clone for SharedMut<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SharedMut<'_, T> {}

// SAFETY : chaque tâche reçoit une partie distincte du tampon (contrat de
// `slice` et de `ptr`) : partager le pointeur revient à envoyer à chaque
// fil un `&mut` sur sa partie, ce que permet `T: Send`.
unsafe impl<T: Send> Send for SharedMut<'_, T> {}
unsafe impl<T: Send> Sync for SharedMut<'_, T> {}

impl<'a, T> SharedMut<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _borrow: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// La partie `[start, start + len)` du tampon, à partager à son tour.
    pub fn view(self, start: usize, len: usize) -> Self {
        assert!(
            start <= self.len && len <= self.len - start,
            "partie [{start}, {start} + {len}) hors d'un tampon de {}",
            self.len
        );
        Self {
            // SAFETY : `start <= len`, le pointeur reste dans le tampon ou
            // juste après.
            ptr: unsafe { self.ptr.add(start) },
            len,
            _borrow: PhantomData,
        }
    }

    /// Un pointeur sur l'élément `offset`, pour un kernel qui écrit un bloc
    /// à partir de là.
    ///
    /// # Safety
    ///
    /// Tout ce qu'on écrit à travers lui doit rester dans le tampon, et
    /// n'appartenir qu'à la tâche qui écrit.
    pub unsafe fn ptr(&self, offset: usize) -> *mut T {
        assert!(
            offset < self.len,
            "élément {offset} hors d'un tampon de {}",
            self.len
        );
        // SAFETY : `offset < len`.
        unsafe { self.ptr.add(offset) }
    }

    /// La tranche `[start, start + len)` du tampon.
    ///
    /// # Safety
    ///
    /// Deux tâches ne doivent jamais recevoir de tranches qui se recouvrent,
    /// ni lire une tranche qu'une autre écrit.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &'a mut [T] {
        assert!(
            start <= self.len && len <= self.len - start,
            "tranche [{start}, {start} + {len}) hors d'un tampon de {}",
            self.len
        );
        // SAFETY : la tranche est dans le tampon, qui vit `'a` ; le partage
        // est le contrat de l'appelant.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once() {
        let pool = Pool::new(4);
        for tasks in [0, 1, 3, 100, 1000] {
            let counts: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(tasks, |task| {
                counts[task].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1)
            );
        }
    }

    #[test]
    fn nested_calls_run_in_place() {
        let pool = Pool::new(3);
        let total = AtomicUsize::new(0);
        pool.run(8, |_| {
            pool.run(5, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 40);
    }

    #[test]
    fn sleeping_workers_wake_up() {
        let pool = Pool::new(4);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let total = AtomicUsize::new(0);
        pool.run(64, |task| {
            total.fetch_add(task, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 63 * 64 / 2);
    }

    #[test]
    #[should_panic(expected = "une tâche du groupe de fils a paniqué")]
    fn a_panicking_task_is_reported() {
        let pool = Pool::new(2);
        pool.run(16, |task| assert_ne!(task, 7));
    }

    #[test]
    fn tasks_write_disjoint_parts() {
        let pool = Pool::new(4);
        let mut data = vec![0usize; 1000];
        let out = SharedMut::new(&mut data);
        pool.run(10, |task| {
            // SAFETY : chaque tâche prend ses 100 éléments.
            let part = unsafe { out.slice(task * 100, 100) };
            for (i, x) in part.iter_mut().enumerate() {
                *x = task * 100 + i;
            }
        });
        assert!(data.iter().enumerate().all(|(i, &x)| i == x));
    }

    #[test]
    #[should_panic(expected = "hors d'un tampon de 10")]
    fn a_slice_beyond_the_buffer_is_refused() {
        let mut data = vec![0u8; 10];
        let out = SharedMut::new(&mut data);
        // SAFETY : une seule tranche ; elle est refusée avant d'exister.
        let _ = unsafe { out.slice(8, 3) };
    }

    #[test]
    fn a_view_is_a_part_of_the_buffer() {
        let mut data = vec![0u8; 10];
        let part = SharedMut::new(&mut data).view(4, 3);
        assert_eq!(part.len(), 3);
        // SAFETY : une seule tranche.
        unsafe { part.slice(0, 3) }.fill(1);
        assert_eq!(data, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "hors d'un tampon de 3")]
    fn a_view_keeps_its_bounds() {
        let mut data = vec![0u8; 10];
        let part = SharedMut::new(&mut data).view(4, 3);
        // SAFETY : une seule tranche ; elle est refusée avant d'exister.
        let _ = unsafe { part.slice(2, 2) };
    }
}
