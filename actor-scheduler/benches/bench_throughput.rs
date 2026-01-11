use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;

struct CountingActor {
    data_count: Arc<AtomicUsize>,
    control_count: Arc<AtomicUsize>,
    mgmt_count: Arc<AtomicUsize>,
}

impl Actor<i32, (), ()> for CountingActor {
    fn handle_data(&mut self, _data: i32) {
        self.data_count.fetch_add(1, Ordering::Relaxed);
    }

    fn handle_control(&mut self, _ctrl: ()) {
        self.control_count.fetch_add(1, Ordering::Relaxed);
    }

    fn handle_management(&mut self, _mgmt: ()) {
        self.mgmt_count.fetch_add(1, Ordering::Relaxed);
    }

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

fn bench_data_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_throughput");

    for message_count in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("messages", message_count),
            &message_count,
            |b, &count| {
                b.iter(|| {
                    let data_count = Arc::new(AtomicUsize::new(0));
                    let control_count = Arc::new(AtomicUsize::new(0));
                    let mgmt_count = Arc::new(AtomicUsize::new(0));

                    let (tx, mut rx) = ActorScheduler::new(1024, 1000);

                    let actor_data = data_count.clone();
                    let actor_control = control_count.clone();
                    let actor_mgmt = mgmt_count.clone();

                    let actor_handle = thread::spawn(move || {
                        let mut actor = CountingActor {
                            data_count: actor_data,
                            control_count: actor_control,
                            mgmt_count: actor_mgmt,
                        };
                        rx.run(&mut actor);
                    });

                    // Send messages
                    for i in 0..count {
                        tx.send(Message::Data(i)).unwrap();
                    }

                    tx.send(Message::Shutdown).unwrap();
                    actor_handle.join().unwrap();

                    let processed = data_count.load(Ordering::Relaxed);
                    black_box(processed)
                });
            },
        );
    }
    group.finish();
}

fn bench_control_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_throughput");

    for message_count in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("messages", message_count),
            &message_count,
            |b, &count| {
                b.iter(|| {
                    let data_count = Arc::new(AtomicUsize::new(0));
                    let control_count = Arc::new(AtomicUsize::new(0));
                    let mgmt_count = Arc::new(AtomicUsize::new(0));

                    let (tx, mut rx) = ActorScheduler::new(1024, 100);

                    let actor_data = data_count.clone();
                    let actor_control = control_count.clone();
                    let actor_mgmt = mgmt_count.clone();

                    let actor_handle = thread::spawn(move || {
                        let mut actor = CountingActor {
                            data_count: actor_data,
                            control_count: actor_control,
                            mgmt_count: actor_mgmt,
                        };
                        rx.run(&mut actor);
                    });

                    // Send control messages
                    for _ in 0..count {
                        tx.send(Message::Control(())).unwrap();
                    }

                    tx.send(Message::Shutdown).unwrap();
                    actor_handle.join().unwrap();

                    let processed = control_count.load(Ordering::Relaxed);
                    black_box(processed)
                });
            },
        );
    }
    group.finish();
}

fn bench_mixed_throughput(c: &mut Criterion) {
    c.bench_function("mixed_lanes_10k_each", |b| {
        b.iter(|| {
            let data_count = Arc::new(AtomicUsize::new(0));
            let control_count = Arc::new(AtomicUsize::new(0));
            let mgmt_count = Arc::new(AtomicUsize::new(0));

            let (tx, mut rx) = ActorScheduler::new(1024, 1000);

            let actor_data = data_count.clone();
            let actor_control = control_count.clone();
            let actor_mgmt = mgmt_count.clone();

            let actor_handle = thread::spawn(move || {
                let mut actor = CountingActor {
                    data_count: actor_data,
                    control_count: actor_control,
                    mgmt_count: actor_mgmt,
                };
                rx.run(&mut actor);
            });

            // Send 10k of each message type in interleaved fashion
            for i in 0..10_000 {
                tx.send(Message::Data(i)).unwrap();
                tx.send(Message::Control(())).unwrap();
                tx.send(Message::Management(())).unwrap();
            }

            tx.send(Message::Shutdown).unwrap();
            actor_handle.join().unwrap();

            let total = data_count.load(Ordering::Relaxed)
                + control_count.load(Ordering::Relaxed)
                + mgmt_count.load(Ordering::Relaxed);
            black_box(total)
        });
    });
}

criterion_group!(
    benches,
    bench_data_throughput,
    bench_control_throughput,
    bench_mixed_throughput
);
criterion_main!(benches);
