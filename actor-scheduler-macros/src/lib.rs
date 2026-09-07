//! Proc macros for actor-scheduler troupe system.
//!
//! This crate provides two macros:
//! - `troupe!` - Generates a Troupe struct with Directory, ExposedHandles, and lifecycle methods
//! - `ports!` - Generates an actor's output word and its wiring

use proc_macro::{Delimiter, TokenStream, TokenTree};

/// Actor attributes parsed from bracket syntax
#[derive(Default)]
struct ActorAttrs {
    is_main: bool,
    is_exposed: bool,
    is_waker: bool,
}

/// Parse attributes from a bracket group like [main], [expose], [main, expose]
fn parse_attrs(group_str: &str) -> ActorAttrs {
    let mut attrs = ActorAttrs::default();
    for part in group_str.split(',') {
        match part.trim() {
            "main" => attrs.is_main = true,
            "expose" => attrs.is_exposed = true,
            // An actor that blocks in `handle_os()` on a file descriptor (epoll,
            // kqueue) needs a `WakeHandler` wired into its inbound handles so
            // sends interrupt the poll. `[waker]` reserves a slot in the
            // generated `Wakers` struct for that actor. `[main]` implies a
            // waker slot too (the platform waker), preserving prior behavior.
            "waker" => attrs.is_waker = true,
            "" => {}
            other => panic!("unknown attribute: {}", other),
        }
    }
    attrs
}

/// Generates a Troupe struct with Directory, ExposedHandles, and lifecycle methods.
///
/// # Syntax
///
/// ```ignore
/// troupe! {
///     actor_name: ActorType,
///     actor_name: ActorType [main],      // runs on calling thread
///     actor_name: ActorType [expose],    // handle exposed to parent
///     actor_name: ActorType [main, expose], // both
/// }
/// ```
///
/// # Example
///
/// ```ignore
/// troupe! {
///     engine: EngineActor [expose],
///     vsync: VsyncActor,
///     display: DisplayActor [main],
/// }
/// ```
///
/// This generates:
/// - `pub struct Directory { ... }` - all actor handles
/// - `pub struct ExposedHandles { ... }` - only [expose] handles
/// - `pub struct Troupe { ... }` - owns schedulers
/// - `impl Troupe`:
///   - `pub fn new() -> Self` - creates channels, builds directory
///   - `pub fn exposed(&self) -> ExposedHandles` - clones exposed handles
///   - `pub fn play(self) -> Result<()>` - runs scoped threads
/// - `pub fn run() -> Result<()>` - convenience function (new + play)
#[proc_macro]
pub fn troupe(input: TokenStream) -> TokenStream {
    // Parse: name: Type [attrs], ...
    // (name, type, is_main, is_exposed)
    let mut actors: Vec<(String, String, bool, bool)> = Vec::new();
    // Parallel to `actors`: true if the actor gets a `Wakers` slot ([main] or
    // [waker]). Kept separate so the existing 4-tuple destructures are
    // untouched.
    let mut waker_slots: Vec<bool> = Vec::new();
    let mut tokens = input.into_iter().peekable();

    while let Some(tok) = tokens.next() {
        // Expect: name (identifier)
        let name = match tok {
            TokenTree::Ident(id) => id.to_string(),
            TokenTree::Punct(_) => continue, // skip commas
            _ => continue,
        };

        // Expect: :
        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => panic!("expected `:` after actor name '{}'", name),
        }

        // Expect: Type (may include generics like Actor<T> or Actor<'a>)
        // Collect all tokens until we hit [attrs], comma, or EOL
        let mut type_tokens = Vec::new();
        loop {
            match tokens.peek() {
                Some(TokenTree::Punct(p)) if p.as_char() == ',' => break,
                Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => break,
                None => break,
                Some(_) => {}
            }
            if let Some(tok) = tokens.next() {
                type_tokens.push(tok);
            }
        }

        let type_name = type_tokens
            .into_iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("");

        if type_name.is_empty() {
            panic!("expected type after colon for actor '{}'", name);
        }

        // Check for [attrs]
        let mut attrs = ActorAttrs::default();
        if let Some(TokenTree::Group(g)) = tokens.peek()
            && g.delimiter() == Delimiter::Bracket
        {
            let inner = g.stream().to_string();
            attrs = parse_attrs(&inner);
            tokens.next(); // consume the bracket group
        }

        waker_slots.push(attrs.is_main || attrs.is_waker);
        actors.push((name, type_name, attrs.is_main, attrs.is_exposed));

        // Skip comma if present
        if let Some(TokenTree::Punct(p)) = tokens.peek()
            && p.as_char() == ','
        {
            tokens.next();
        }
    }

    // Validate exactly one main
    let main_count = actors.iter().filter(|(_, _, m, _)| *m).count();
    if main_count != 1 {
        panic!(
            "exactly one actor must be marked [main], found {}",
            main_count
        );
    }

    // Generate Directory fields (all actors)
    let dir_fields: String = actors
        .iter()
        .map(|(name, ty, _, _)| {
            format!(
                "pub {name}: ::actor_scheduler::ActorHandle<
                    <{ty} as ::actor_scheduler::ActorTypes>::Data,
                    <{ty} as ::actor_scheduler::ActorTypes>::Control,
                    <{ty} as ::actor_scheduler::ActorTypes>::Management,
                >,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate ExposedHandles fields (only exposed actors)
    let exposed_actors: Vec<_> = actors.iter().filter(|(_, _, _, e)| *e).collect();
    let exposed_fields: String = exposed_actors
        .iter()
        .map(|(name, ty, _, _)| {
            format!(
                "pub {name}: ::actor_scheduler::ActorHandle<
                    <{ty} as ::actor_scheduler::ActorTypes>::Data,
                    <{ty} as ::actor_scheduler::ActorTypes>::Control,
                    <{ty} as ::actor_scheduler::ActorTypes>::Management,
                >,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate exposed() impl - creates new SPSC handles from builders
    let exposed_add_producer: String = exposed_actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}: self.{name}_builder.add_producer(),"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate builder fields for Troupe struct (builders, not schedulers)
    let builder_fields: String = actors
        .iter()
        .map(|(name, ty, _, _)| {
            format!(
                "{name}_builder: ::actor_scheduler::ActorBuilder<
                    <{ty} as ::actor_scheduler::ActorTypes>::Data,
                    <{ty} as ::actor_scheduler::ActorTypes>::Control,
                    <{ty} as ::actor_scheduler::ActorTypes>::Management,
                >,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate per-actor directory fields in Troupe
    let dir_storage_fields: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}_dir: Directory,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate builder creation in new_with_wakers().
    // Waker-slot actors ([main] or [waker]) take their `wakers.<name>` field;
    // all others get None.
    let create_builders: String = actors
        .iter()
        .zip(waker_slots.iter())
        .map(|((name, ty, _, _), is_slot)| {
            let waker = if *is_slot {
                format!("wakers.{name}")
            } else {
                "None".to_string()
            };
            format!(
                "let mut {name}_builder = ::actor_scheduler::ActorBuilder::<
                    <{ty} as ::actor_scheduler::ActorTypes>::Data,
                    <{ty} as ::actor_scheduler::ActorTypes>::Control,
                    <{ty} as ::actor_scheduler::ActorTypes>::Management,
                >::new(1024, {waker});"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // `Wakers` struct: one optional handle per waker slot. Always has at least
    // the [main] actor's field, so `new_with_waker`/`new` shims below compile.
    let wakers_fields: String = actors
        .iter()
        .zip(waker_slots.iter())
        .filter(|(_, is_slot)| **is_slot)
        .map(|((name, _, _, _), _)| {
            format!(
                "pub {name}: ::std::option::Option<::std::sync::Arc<dyn ::actor_scheduler::WakeHandler>>,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // `Wakers { main: main_waker, <other slots>: None }` for the back-compat
    // `new_with_waker` shim.
    let wakers_from_main: String = actors
        .iter()
        .zip(waker_slots.iter())
        .filter(|(_, is_slot)| **is_slot)
        .map(|((name, _, is_main, _), _)| {
            if *is_main {
                format!("{name}: main_waker,")
            } else {
                format!("{name}: None,")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // `Wakers { all slots: None }` for the `new` shim.
    let wakers_all_none: String = actors
        .iter()
        .zip(waker_slots.iter())
        .filter(|(_, is_slot)| **is_slot)
        .map(|((name, _, _, _), _)| format!("{name}: None,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate per-actor directory creation
    // Each actor gets its own Directory with dedicated SPSC handles
    let create_dirs: String = actors
        .iter()
        .map(|(actor_name, _, _, _)| {
            let fields: String = actors
                .iter()
                .map(|(target_name, _, _, _)| {
                    format!("{target_name}: {target_name}_builder.add_producer(),")
                })
                .collect::<Vec<_>>()
                .join("\n                    ");
            format!(
                "let {actor_name}_dir = Directory {{
                    {fields}
                }};"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate Troupe struct init
    let troupe_init: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}_builder, {name}_dir,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate spawns for non-main actors in play()
    let build_schedulers: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("let mut {name}_s = self.{name}_builder.build();"))
        .collect::<Vec<_>>()
        .join("\n");

    let move_dirs: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("let {name}_dir = self.{name}_dir;"))
        .collect::<Vec<_>>()
        .join("\n");

    let spawns: String = actors
        .iter()
        .filter(|(_, _, is_main, _)| !is_main)
        .map(|(name, ty, _, _)| {
            format!(
                r#"
                s.spawn(move || {{
                    let mut actor = <{ty} as ::actor_scheduler::TroupeActor<Directory>>::new({name}_dir);
                    {name}_s.run(&mut actor);
                }});
                "#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate main actor run
    let (main_name, main_ty, _, _) = actors.iter().find(|(_, _, m, _)| *m).unwrap();
    let main_run = format!(
        r#"
        let mut actor = <{main_ty} as ::actor_scheduler::TroupeActor<Directory>>::new({main_name}_dir);
        {main_name}_s.run(&mut actor);
        "#
    );

    // Generate shutdown method - sends in reverse declaration order (last started = first stopped)
    let shutdown_impl = actors
        .iter()
        .rev()
        .map(|(name, _, _, _)| {
            format!("let _ = self.{name}.send(::actor_scheduler::Message::Shutdown);")
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"
        /// Directory containing handles to all actors in this troupe.
        ///
        /// With SPSC channels, each actor gets its OWN Directory instance
        /// where each handle is a dedicated SPSC channel to the target actor.
        pub struct Directory {{
            {dir_fields}
        }}

        impl Directory {{
            /// Initiate graceful shutdown of all actors in this troupe.
            ///
            /// Sends `Message::Shutdown` to each actor in reverse declaration order.
            /// Uses this handle's doorbell (MPSC) for each target actor.
            pub fn shutdown(&self) {{
                {shutdown_impl}
            }}
        }}

        /// Handles exposed to parent troupes.
        pub struct ExposedHandles {{
            {exposed_fields}
        }}

        /// Wake handlers for the troupe's waker-slot actors ([main] / [waker]).
        ///
        /// One optional field per slot: `Some(waker)` wires that `WakeHandler`
        /// into every inbound handle of the actor, so a send interrupts the
        /// `handle_os()` poll it is blocked in; `None` leaves it doorbell-only.
        #[derive(Default)]
        pub struct Wakers {{
            {wakers_fields}
        }}

        /// Troupe manages actor group lifecycle.
        ///
        /// Stores [`ActorBuilder`]s until `play()` so that `exposed()` can
        /// register additional producers. Builders are consumed by `play()`
        /// which seals the registries and spawns the actor threads.
        pub struct Troupe {{
            {builder_fields}
            {dir_storage_fields}
        }}

        impl Troupe {{
            /// Create a new troupe, wiring a wake handler into each waker slot.
            ///
            /// This is phase 1 of two-phase initialization:
            /// 1. `new_with_wakers()` - create builders and per-actor directories
            /// 2. `exposed()` - (optional) create handles for parent troupe
            /// 3. `play()` - build schedulers, spawn threads, run actors
            pub fn new_with_wakers(wakers: Wakers) -> Self {{
                {create_builders}

                // Each actor gets its own Directory with dedicated SPSC handles
                {create_dirs}

                Self {{
                    {troupe_init}
                }}
            }}

            /// Create a new troupe with a wake handler for the [main] actor only.
            ///
            /// Back-compat shim: all other waker slots get `None`.
            pub fn new_with_waker(main_waker: Option<::std::sync::Arc<dyn ::actor_scheduler::WakeHandler>>) -> Self {{
                Self::new_with_wakers(Wakers {{
                    {wakers_from_main}
                }})
            }}

            /// Create a new troupe without any wake handlers.
            pub fn new() -> Self {{
                Self::new_with_wakers(Wakers {{
                    {wakers_all_none}
                }})
            }}

            /// Create exposed handles by adding new SPSC producers.
            ///
            /// Each call creates a fresh set of handles with dedicated channels.
            /// Must be called before `play()` (which consumes the builders).
            pub fn exposed(&mut self) -> ExposedHandles {{
                ExposedHandles {{
                    {exposed_add_producer}
                }}
            }}

            /// Run the troupe. Builds schedulers (sealing all producers),
            /// spawns threads for non-main actors, runs main actor on calling thread.
            pub fn play(self) -> ::std::result::Result<(), ::std::boxed::Box<dyn ::std::error::Error + Send + Sync>> {{
                // Build schedulers — seals the builders, no more producers
                {build_schedulers}

                // Move per-actor directories out of self
                {move_dirs}

                ::std::thread::scope(|s| {{
                    {spawns}

                    {main_run}

                    Ok(())
                }})
            }}
        }}

        /// Convenience function: creates troupe and runs it.
        pub fn run() -> ::std::result::Result<(), ::std::boxed::Box<dyn ::std::error::Error + Send + Sync>> {{
            Troupe::new().play()
        }}
        "#,
    )
    .parse()
    .expect("failed to parse generated troupe code")
}

// ────────────────────────────────────────────────────────────────────────────
// ports! — the output word and its wiring
// ────────────────────────────────────────────────────────────────────────────

/// How one output port is delivered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PortKind {
    /// Parks the actor when the target is full. The only kind that reaches the wiring — every
    /// port parks, so this is also every port's default.
    Blocking,
    /// Addressed to the actor itself. Never reaches the wiring — the scheduler lifts it into
    /// the continuation slot.
    SelfPort,
}

/// One declared port: field name, message type, delivery.
struct Port {
    name: String,
    ty: String,
    kind: PortKind,
}

/// Parse `[self]`, defaulting to a blocking port.
///
/// `[drop]` is a named, deliberate compile-time panic rather than silently accepted or a
/// confusing parse error: droppable ports have no production user, the one real shedding
/// instance in the runtime was a bypass of this very macro (a hand-rolled `try_send`, not a
/// declared `[drop]` port), and every port now parks (see
/// `docs/designs/actor-scheduler-mealy-transducer.md` §7.2).
fn parse_port_kind(group_str: &str, port: &str) -> PortKind {
    let mut kind = PortKind::Blocking;
    for part in group_str.split(',') {
        match part.trim() {
            "drop" => panic!(
                "ports! `{port}`: droppable ports are gone; every port parks. \
                 See docs/designs/actor-scheduler-mealy-transducer.md §7.2."
            ),
            "self" => kind = PortKind::SelfPort,
            "" => {}
            other => {
                panic!("unknown port attribute `{other}` on port `{port}` (expected `self`)")
            }
        }
    }
    kind
}

/// Generates an actor's output word and the wiring that delivers it.
///
/// A Mealy actor returns its output instead of sending it, so every actor needs two types
/// that are pure boilerplate: an output word with one optional slot per downstream port, and
/// a wiring that delivers each set port and **leaves blocked ones in place**. Writing them by
/// hand is repetitive and quietly dangerous — a port omitted from `flush` is a message that
/// vanishes with no error anywhere.
///
/// # Syntax
///
/// ```ignore
/// ports! {
///     App {
///         engine: Render,        // parks the actor when full — the only delivery kind
///         write: Echo,
///         again: u32 [self],     // continuation: goes to the slot, not the wiring
///     }
/// }
/// ```
///
/// # Generates
///
/// - `pub struct AppOut` — one `Option<T>` per port, `#[derive(Default)]` for the silent
///   transition.
/// - `pub struct AppWiring` — one `SpscSender<T>` per *deliverable* port. A `[self]` port has
///   no field here, because it never travels through the wiring.
/// - `impl Wiring for AppWiring` — `flush` delivering each port via `mealy::send_port` and
///   combining the outcomes, so a partial flush falls out: the ports that fit go now, the rest
///   wait.
/// - `impl AppOut::take_continuation` when a `[self]` port is declared, for the actor's
///   `Transducer::take_continuation` to delegate to.
///
/// # Panics
///
/// At expansion time, on malformed syntax, an unknown port attribute, or more than one
/// `[self]` port — the continuation slot holds exactly one message.
#[proc_macro]
pub fn ports(input: TokenStream) -> TokenStream {
    let mut tokens = input.into_iter().peekable();

    let base = match tokens.next() {
        Some(TokenTree::Ident(id)) => id.to_string(),
        other => panic!("ports!: expected an actor name, found {other:?}"),
    };

    let body = match tokens.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g.stream(),
        _ => panic!("ports!: expected `{{ ... }}` after `{base}`"),
    };

    let ports = parse_ports(body, &base);

    let self_ports: Vec<&Port> = ports
        .iter()
        .filter(|p| p.kind == PortKind::SelfPort)
        .collect();
    assert!(
        self_ports.len() <= 1,
        "ports!: `{base}` declares {} `[self]` ports, but the continuation slot holds exactly one",
        self_ports.len()
    );

    render_ports(&base, &ports, self_ports.first().copied())
}

/// Parse the `name: Type [attrs],` list inside the braces.
fn parse_ports(body: TokenStream, base: &str) -> Vec<Port> {
    let mut ports: Vec<Port> = Vec::new();
    let mut tokens = body.into_iter().peekable();

    while let Some(tok) = tokens.next() {
        let name = match tok {
            TokenTree::Ident(id) => id.to_string(),
            TokenTree::Punct(_) => continue, // trailing / separating commas
            other => panic!("ports! `{base}`: expected a port name, found {other:?}"),
        };

        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => panic!("ports! `{base}`: expected `:` after port `{name}`"),
        }

        // The type runs until a comma or an attribute bracket, so generics pass through.
        let mut ty_tokens = Vec::new();
        loop {
            match tokens.peek() {
                Some(TokenTree::Punct(p)) if p.as_char() == ',' => break,
                Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => break,
                None => break,
                Some(_) => {}
            }
            if let Some(tok) = tokens.next() {
                ty_tokens.push(tok);
            }
        }
        let ty = ty_tokens
            .into_iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("");
        assert!(
            !ty.is_empty(),
            "ports! `{base}`: expected a type after `{name}:`"
        );

        let mut kind = PortKind::Blocking;
        if let Some(TokenTree::Group(g)) = tokens.peek()
            && g.delimiter() == Delimiter::Bracket
        {
            kind = parse_port_kind(&g.stream().to_string(), &name);
            tokens.next();
        }

        ports.push(Port { name, ty, kind });
    }

    assert!(!ports.is_empty(), "ports! `{base}`: declares no ports");
    ports
}

/// Emit the output word, the wiring, and their impls.
fn render_ports(base: &str, ports: &[Port], self_port: Option<&Port>) -> TokenStream {
    let out_name = format!("{base}Out");
    let wiring_name = format!("{base}Wiring");

    let out_fields = ports
        .iter()
        .map(|p| {
            format!(
                "    /// The `{}` port.\n    pub {}: ::core::option::Option<{}>,",
                p.name, p.name, p.ty
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let deliverable: Vec<&Port> = ports
        .iter()
        .filter(|p| p.kind != PortKind::SelfPort)
        .collect();

    let wiring_fields = deliverable
        .iter()
        .map(|p| {
            format!(
                "    /// Target of the `{}` port.\n    pub {}: ::actor_scheduler::spsc::SpscSender<{}>,",
                p.name, p.name, p.ty
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let sends = deliverable
        .iter()
        .map(|p| {
            format!(
                "            ::actor_scheduler::mealy::send_port(&mut out.{0}, &self.{0}),",
                p.name
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // A continuation must have been lifted into the slot before the wiring runs. Assert it
    // rather than silently delivering nowhere.
    let self_guard = self_port.map_or_else(String::new, |p| {
        format!(
            "        debug_assert!(\n            out.{}.is_none(),\n            \"the `{}` port is a continuation and must not reach the wiring\"\n        );\n",
            p.name, p.name
        )
    });

    let continuation_impl = self_port.map_or_else(String::new, |p| {
        format!(
            r#"
impl {out_name} {{
    /// Take the `{port}` continuation, for `Transducer::take_continuation` to delegate to.
    ///
    /// The scheduler calls this before the wiring runs, so a self-addressed message never
    /// touches a queue and can never block.
    #[must_use]
    pub fn take_continuation(&mut self) -> ::core::option::Option<{ty}> {{
        self.{port}.take()
    }}
}}
"#,
            out_name = out_name,
            port = p.name,
            ty = p.ty
        )
    });

    format!(
        r#"
/// Output word for `{base}`: one optional slot per port.
///
/// `Default` is the silent transition — every port unset.
#[derive(Default)]
pub struct {out_name} {{
{out_fields}
}}

/// Where `{base}`'s output ports go.
pub struct {wiring_name} {{
{wiring_fields}
}}

impl ::actor_scheduler::mealy::Wiring for {wiring_name} {{
    type Out = {out_name};

    fn flush(&mut self, out: &mut {out_name}) -> ::actor_scheduler::mealy::Flush {{
{self_guard}        ::actor_scheduler::mealy::all([
{sends}
        ])
    }}
}}
{continuation_impl}"#,
    )
    .parse()
    .expect("ports!: failed to parse generated code")
}
