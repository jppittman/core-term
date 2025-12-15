use super::traits::{Application, EventHandler, Window};
use crate::api::private::{DisplayEvent, DriverCommand, WindowId};
use crate::api::public::WindowDescriptor;
use crate::channel::EngineSender;
use crate::display::driver::DisplayDriver;
use anyhow::Result;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::mpsc::TryRecvError;

/// Generic Display Driver.
/// A = Application, W = Window (via A::Window)
pub struct Driver<A: Application> {
    // Only kept for the main thread run/init
    // But `DisplayDriver` trait requires `Clone` + `Send`.
    // The `Driver` struct *is* the thing we clone around. but it can't hold `A` (which is !Send usually and !Clone).
    // Ah, the `DisplayDriver` trait pattern implies `new` creates the channels, and `run` creates the App.
    // So `Driver` struct here is just the "Shell".

    // We need a phantom data or configuration to know which A to create in run().
    // The `DisplayDriver` trait says:
    // fn new(engine_tx) -> Result<Self>
    // fn send(&self, cmd) ...
    // fn run(&self) ...

    // So `Driver` logic needs to manage specific channels.
    // But wait, `DisplayDriver` trait definition:
    // pub trait DisplayDriver: Clone + Send { ... }

    // So `Driver` struct must be Clone + Send. It holds the sender to the "Real Driver" (the event loop).
    // The `run()` method consumes the receiver?

    // Actually the current `driver.rs` says:
    // - driver.send() queues commands.
    // - driver.run() READS commands and executes them.

    // So `Driver` generic needs to hold:
    // 1. The receiver (guarded by Mutex? Or strictly, run() takes &self but is blocking unique).
    //    Actually `run(&self)` is weird if it consumes the receiver.
    //    The current `DisplayDriver` trait signature `run(&self)` is indeed a bit conflicting with ownership if we have a Receiver.
    //    But assuming `run` is called once.

    // Let's implement it with internal mutability or just `Mutex<Receiver>`.
    cmd_tx: std::sync::mpsc::Sender<DriverCommand<A::Pixel>>,
    cmd_rx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<DriverCommand<A::Pixel>>>>,
    engine_tx: EngineSender<A::Pixel>,

    _phantom: PhantomData<A>,
}

unsafe impl<A: Application> Send for Driver<A> {}

impl<A: Application> Clone for Driver<A> {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            cmd_rx: self.cmd_rx.clone(),
            engine_tx: self.engine_tx.clone(),
            _phantom: PhantomData,
        }
    }
}

// Ensure A::Pixel acts as the P
impl<A: Application> DisplayDriver for Driver<A>
where
    A::Pixel: 'static,
{
    type Pixel = A::Pixel;

    fn new(engine_tx: EngineSender<Self::Pixel>) -> Result<Self> {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        Ok(Self {
            cmd_tx,
            cmd_rx: std::sync::Arc::new(std::sync::Mutex::new(cmd_rx)),
            engine_tx,
            _phantom: PhantomData,
        })
    }

    fn send(&self, cmd: DriverCommand<Self::Pixel>) -> Result<()> {
        self.cmd_tx.send(cmd)?;
        // If we need to wake up the loop, assuming `run` handles it via polling or waker?
        // The implementation plan says "Application::run" pumps events.
        // We might need a Waker mechanism.
        // For now, let's assume the platform implementation handles waking or polling.
        // Actually, `Waker` is platform specific.
        // We might need `A::waker() -> Waker`?
        // Let's stick thereto for now.
        Ok(())
    }

    fn run(&self) -> Result<()> {
        let mut app = A::new()?;
        // Windows map
        let mut windows: HashMap<WindowId, A::Window> = HashMap::new();

        // The Event Handler that bridges App -> Engine
        struct Bridge<'a, A: Application> {
            engine_tx: &'a EngineSender<A::Pixel>,
            windows: &'a mut HashMap<WindowId, A::Window>,
            app: &'a A,
            cmd_rx: &'a std::sync::mpsc::Receiver<DriverCommand<A::Pixel>>,
            running: bool,
        }

        impl<'a, A: Application> EventHandler for Bridge<'a, A> {
            fn handle_event(&mut self, event: DisplayEvent) {
                // Forward to engine
                // DisplayEvent -> Message conversion is handled by From impl
                let _ = self.engine_tx.send(event.into());
            }
        }

        // Wait, `A::run` takes `&self` and `handler`. It blocks.
        // How do we process `cmd_rx`?
        // The `handler` is called by the platform loop.
        // BUT the platform loop usually blocks!
        // So we need a way to process commands inside the loop.
        // Standard pattern: Platform loop calls a "User callback" periodically or on wake.
        // OR `Application::run` is responsible for polling `handler`? No, handler is passive.

        // This suggests `Application` trait needs a way to let us inject logic,
        // OR `EventHandler` should have a `tick()` or `poll()`?

        // Let's refine `EventHandler`:
        // trait EventHandler { fn handle_event(...); }
        // We probably need `trait Application { fn run<F>(&self, callback: F) }` ?

        // Actually, if `A::run` blocks, it must support waking up to process our commands.
        // So we need to poll cmd_rx inside the `handler` call?
        // Note: `handle_event` only triggers on OS events.
        // If we send a command, we wake the loop (via Waker, which we haven't defined in trait yet, oops).

        // The user comment "How do you create a window without talking to the application layer?"
        // Implies interaction.

        // Let's assume for this step we define the structure,
        // and we will add `Waker` to `Application` trait if needed, or rely on `A::run` calling us back.
        // Many loops have `on_idle`.

        // Simplified approach:
        // `A::run` takes `&mut dyn EventHandler`.
        // We wrap our command logic into the EventHandler?
        // But `GenericDriver::run` needs to define that handler.

        // Okay, let's write `run` logic assuming `A::run` drives the show.

        let rx_lock = self.cmd_rx.lock().unwrap(); // Lock for duration of run?
                                                   // No, `run` is long lived.
                                                   // We shouldn't hold the lock if we wanted others to clone the driver?
                                                   // But `DisplayDriver` says `run()` is the consumer. Only one `run` call.
                                                   // So holding lock is fine or consuming it.

        let rx = &*rx_lock; // Borrow the receiver.

        let mut handler = Handler {
            engine_tx: &self.engine_tx,
            windows: &mut windows,
            app: &app,
            cmd_rx: rx,
            should_quit: false,
        };

        // Run the app!
        app.run(&mut handler)?;

        Ok(())
    }
}

struct Handler<'a, A: Application> {
    engine_tx: &'a EngineSender<A::Pixel>,
    windows: &'a mut HashMap<WindowId, A::Window>,
    app: &'a A,
    cmd_rx: &'a std::sync::mpsc::Receiver<DriverCommand<A::Pixel>>,
    should_quit: bool,
}

impl<'a, A: Application> EventHandler for Handler<'a, A> {
    fn handle_event(&mut self, event: DisplayEvent) {
        // Forward event
        let _ = self
            .engine_tx
            .send(crate::channel::EngineData::FromDriver(event));

        // Process commands whenever we get an event (or wake up)
        // This assumes we get a "Wake" event or frequent polling.
        // We might need to drain the command queue here.
        loop {
            match self.cmd_rx.try_recv() {
                Ok(cmd) => self.process_command(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.should_quit = true;
                    break;
                }
            }
        }
    }
}

impl<'a, A: Application> Handler<'a, A> {
    fn process_command(&mut self, cmd: DriverCommand<A::Pixel>) {
        match cmd {
            DriverCommand::CreateWindow {
                id,
                width,
                height,
                title,
            } => {
                let desc = WindowDescriptor {
                    width,
                    height,
                    title,
                    ..Default::default()
                };
                match self.app.create_window(id, desc) {
                    Ok(win) => {
                        self.windows.insert(id, win);
                        // Convert A::Pixel info to DisplayEvent?
                        // WindowCreated is sent by the Window implementation or us?
                        // Let's send it here to be safe/consistent.
                        let _ = self.engine_tx.send(
                            DisplayEvent::WindowCreated {
                                id,
                                width_px: width,
                                height_px: height,
                                scale: 1.0, // TODO: Get from window
                            }
                            .into(),
                        );
                    }
                    Err(e) => eprintln!("Failed to create window: {}", e),
                }
            }
            DriverCommand::SetTitle { id, title } => {
                if let Some(win) = self.windows.get_mut(&id) {
                    win.set_title(&title);
                }
            }
            DriverCommand::SetSize { id, width, height } => {
                // win.set_size...
            }
            DriverCommand::DestroyWindow { id } => {
                self.windows.remove(&id);
                // Window Drop impl should close it.
            }
            DriverCommand::Shutdown => {
                // Signal app to quit?
                // We typically need an `app.quit()` or return from run.
                // Depending on platform.
                // For now, we can set a flag if the loop supports it.
            }
            // ... Copy, Paste, Bell ...
            _ => {}
        }
    }
}
