# pixelflow-runtime/src/api

**Is.** The vocabulary an application and the engine speak across the pull: the
engine tells the application the lattice's extent (`WindowCreated`, `Resized`;
a resize is a new extent and a recompile, DENOTATION §5) and asks for a frame;
the application answers with a `Scene`, the compiled program that collapses
into that frame, or with `Skipped`. `public.rs` is what an application sees;
`private.rs` is what the engine's own actors exchange. Types, no behavior.

## Owns
- The application's lanes, by type: `EngineEvent` out (`Control`, `Management`,
  `Data`), `AppManagement` and `AppData` in. Which lane a message rides is
  decided once, by which enum holds it.
- The engine's own message set, handle types, burst and buffer sizes.
- Registration order: `UnregisteredEngineHandle::register` consumes the handle,
  sends `RegisterApp` then `CreateWindow`, and is the only way (bar a
  `#[doc(hidden)]` test constructor) to an `EngineHandle`.

## Never
- No decision: nothing here chooses a lane, a window size or a frame. The only
  bodies are `Debug`, `From`, `Application::send` and `register`.
- No send result dropped: `register` and `Application::send` return a `RuntimeError`.
- No terminal vocabulary (PTY, ANSI, cell), no dependency on `core-term`.
- No lane, width, `std::arch` or platform `#[cfg]`: a `Scene` arrives compiled.
- No coordinate transform on a `Scene`: device pixels by construction.

## Contract
`mod.rs` declares the two modules; `lib.rs` re-exports `public::*`. Each
variant's doc carries its contract (**Engine** / **Application**).
- `public.rs`: `EngineEvent` (`EngineEventControl`, `EngineEventManagement`,
  `EngineEventData`), `AppData`, `AppManagement`, `Application`,
  `UnregisteredEngineHandle`, `EngineHandle`, `WindowDescriptor`, `WindowId`,
  `CursorIcon`.
- `private.rs`: `EngineData`, `EngineControl`, `GreenReadyBundle`, `DriverCommand`,
  `EngineActorHandle`, `EngineActorScheduler`, `create_engine_actor`.

## Enforced by
- Types: `UnregisteredEngineHandle::new` is `pub(crate)` and `register` takes
  `self`, so a frame cannot be sent before registration; `GreenReadyBundle`
  has named fields, so `host` and `forwarder` (one type) cannot be transposed.
- desloppify `actor-lane-choice` (covers `pixelflow-runtime/**`);
  `scripts/check-cfg-encapsulation.sh`, no `api/` entry in its baseline.
- `pixelflow-runtime/src/engine_core.rs` tests pin what `AppData` does.

## Behind
- `private` is `pub mod`: `core-term/src/terminal_app.rs` and
  `pixelflow-runtime/tests/platform_actor_tests.rs` name
  `api::private::EngineActorScheduler` and `create_engine_actor`.
- `DriverCommand` is the legacy driver vocabulary: only the deprecated
  `DisplayDriver` trait (`display/driver.rs`) and the web driver, which does not
  build, use it; live drivers take `DisplayData`/`DisplayControl`/`DisplayMgmt`.
- `AppData::Skipped`'s doc says the engine may present a blank frame; it only
  returns the vsync token (`EngineCore::step_app_data`).
- `EngineHandler` still relays app input to the coordinator; this API follows
  the mesh (mesh-migration design, Draft 2026-07-25).

## Decided in
`docs/designs/pixelflow-runtime-engine-mesh-migration.md` (Draft),
`docs/designs/actor-scheduler-mealy-transducer.md` (Draft); CLAUDE.md "Actor Model".
