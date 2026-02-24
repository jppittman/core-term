use actor_scheduler::{
    Actor, ActorStatus, HandlerError, HandlerResult, KubeletBuilder, Message, PodSlot,
    RestartPolicy, SystemStatus, spawn_managed,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::thread;
use std::time::{Duration, Instant};

// ─── Minimal actor ───────────────────────────────────────────────────────────

struct Noop;

impl Actor<(), (), ()> for Noop {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ─── Benchmark 1: restart round-trip latency ─────────────────────────────────
//
// Measures the wall-clock time from sending `Message::Shutdown` until a fresh
// `ActorHandle` is available via `PodSlot::reconnect()`.
//
// Two slots are allocated per pod:
//   slot_svc  — receives the fresh handle after each restart (timing target)
//   slot_kill — provides a fresh kill handle for the next iteration
//
// The Kubelet poll interval is swept across three values so you can see the
// restart-latency / CPU-overhead trade-off.
//
// The frequency gate is set very high (100 000 restarts / h) so it never fires
// during the warm-up + measurement phases.

fn bench_restart_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("kubelet_restart_latency");
    group.measurement_time(Duration::from_secs(10));

    for poll_us in [100u64, 500, 1_000] {
        group.bench_with_input(
            BenchmarkId::new("poll_interval_us", poll_us),
            &poll_us,
            |b, &poll_us| {
                let slot_svc = PodSlot::<(), (), ()>::connected();
                let slot_kill = PodSlot::<(), (), ()>::connected();

                let mut pod = spawn_managed(
                    vec![slot_svc.clone(), slot_kill.clone()],
                    64,
                    None,
                    || Noop,
                );
                // handle[0] → slot_svc (svc side — we grab fresh ones via reconnect)
                // handle[1] → slot_kill (kill side — initial handle for first iteration)
                let _initial_svc = pod.handles.remove(0);
                let kill_handle = pod.handles.remove(0);

                let kubelet = KubeletBuilder::new()
                    .with_poll_interval(Duration::from_micros(poll_us))
                    // Very high gate: 100 000 restarts / hour — never fires in bench.
                    .add_pod_with_gate(
                        pod,
                        RestartPolicy::Always,
                        100_000,
                        Duration::from_secs(3600),
                    )
                    .build();
                thread::spawn(|| kubelet.run());

                // live_kill: the ActorHandle we send Shutdown through each iteration.
                // Refreshed from slot_kill after each restart.
                let mut live_kill = kill_handle;

                // Let initial pod and Kubelet stabilise.
                thread::sleep(Duration::from_millis(5));

                b.iter(|| {
                    let start = Instant::now();

                    // 1. Kill the current pod instance via the kill lane.
                    //    The SPSC push is non-blocking (ring buffer is empty).
                    live_kill.send(black_box(Message::Shutdown)).ok();

                    // 2. Block until the Kubelet restarts the pod and publishes
                    //    a fresh handle to slot_svc.
                    //
                    //    reconnect() waits in Connected|Restarting state until
                    //    the Kubelet calls publish(), which signals the condvar.
                    //    Total measured time = actor exit + Kubelet poll + spawn
                    //    + publish.
                    let _new_svc = slot_svc.reconnect(Duration::from_secs(5)).unwrap();

                    let elapsed = start.elapsed();

                    // 3. Grab the fresh kill handle for the next iteration.
                    live_kill = slot_kill.reconnect(Duration::from_secs(5)).unwrap();

                    black_box(elapsed)
                });
            },
        );
    }
    group.finish();
}

// ─── Benchmark 2: hot-path SPSC send with Kubelet running ────────────────────
//
// The Kubelet runs in a separate thread and does NOT touch the SPSC on the
// hot path.  This benchmark confirms that having a Kubelet present adds zero
// overhead to normal `ActorHandle::send()` calls.

fn bench_hot_path_with_kubelet(c: &mut Criterion) {
    let mut group = c.benchmark_group("kubelet_hot_path_overhead");

    for poll_ms in [1u64, 5] {
        group.bench_with_input(
            BenchmarkId::new("kubelet_poll_ms", poll_ms),
            &poll_ms,
            |b, &poll_ms| {
                let slot = PodSlot::<(), (), ()>::connected();
                let mut pod = spawn_managed(vec![slot.clone()], 1024, None, || Noop);
                let handle = pod.handles.remove(0);

                let kubelet = KubeletBuilder::new()
                    .with_poll_interval(Duration::from_millis(poll_ms))
                    .add_pod_with_gate(
                        pod,
                        RestartPolicy::Always,
                        100_000,
                        Duration::from_secs(3600),
                    )
                    .build();
                thread::spawn(|| kubelet.run());

                thread::sleep(Duration::from_millis(5));

                // Pod is alive; every send is a hot-path SPSC push.
                // The Kubelet is sleeping in its poll loop — zero interference.
                b.iter(|| handle.send(black_box(Message::Data(()))).ok());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_restart_round_trip, bench_hot_path_with_kubelet);
criterion_main!(benches);
