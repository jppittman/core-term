//! Proc macros for actor-scheduler troupe system.
//!
//! This crate provides two macros:
//! - `#[actor_impl]` - Transforms an impl block into an Actor trait impl
//! - `troupe!` - Generates a Troupe struct with Directory, ExposedHandles, and lifecycle methods

use proc_macro::{Delimiter, TokenStream, TokenTree};

/// Transforms an impl block into an Actor trait implementation.
///
/// # Example
///
/// ```ignore
/// #[actor_impl]
/// impl EngineActor<'_> {
///     type Data = EngineData;
///     type Control = EngineControl;
///     type Management = EngineManagement;
///
///     fn new(dir: &Directory) -> Self { Self { dir } }
///     fn handle_data(&mut self, msg: Self::Data) { }
///     fn handle_control(&mut self, msg: Self::Control) { }
///     fn handle_management(&mut self, msg: Self::Management) { }
/// }
/// ```
///
/// Generates:
///
/// ```ignore
/// impl<'__dir, __Dir> TroupeActor<'__dir, __Dir> for EngineActor<'__dir>
/// where
///     __Dir: '__dir,
/// {
///     // ... body
/// }
/// ```
#[proc_macro_attribute]
pub fn actor_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut tokens = item.into_iter().peekable();

    // Find type name after `impl`
    let mut type_name: Option<String> = None;

    while let Some(tok) = tokens.next() {
        match tok {
            TokenTree::Ident(id) if id.to_string() == "impl" => {
                // Look for the type name, skipping any lifetime
                while let Some(tok) = tokens.next() {
                    match tok {
                        TokenTree::Ident(id) => {
                            type_name = Some(id.to_string());
                            break;
                        }
                        TokenTree::Punct(p) if p.as_char() == '<' => {
                            // Skip <'_> or <'a>
                            let mut depth = 1;
                            while depth > 0 {
                                match tokens.next() {
                                    Some(TokenTree::Punct(p)) if p.as_char() == '<' => depth += 1,
                                    Some(TokenTree::Punct(p)) if p.as_char() == '>' => depth -= 1,
                                    None => panic!("unexpected end in lifetime"),
                                    _ => {}
                                }
                            }
                        }
                        _ => continue,
                    }
                }
                break;
            }
            _ => continue,
        }
    }

    let type_name = type_name.expect("#[actor_impl] must be on impl block with type name");

    // Find body brace
    let mut body: Option<String> = None;
    while let Some(tok) = tokens.next() {
        if let TokenTree::Group(g) = tok {
            if g.delimiter() == Delimiter::Brace {
                body = Some(g.stream().to_string());
                break;
            }
        }
    }

    let body = body.expect("no impl body found");

    format!(
        r#"
        impl<'__dir, __Dir> ::actor_scheduler::TroupeActor<'__dir, __Dir> for {type_name}<'__dir>
        where
            __Dir: '__dir,
        {{
            {body}
        }}
        "#
    )
    .parse()
    .expect("failed to parse generated impl")
}

/// Actor attributes parsed from bracket syntax
#[derive(Default)]
struct ActorAttrs {
    is_main: bool,
    is_exposed: bool,
}

/// Parse attributes from a bracket group like [main], [expose], [main, expose]
fn parse_attrs(group_str: &str) -> ActorAttrs {
    let mut attrs = ActorAttrs::default();
    for part in group_str.split(',') {
        match part.trim() {
            "main" => attrs.is_main = true,
            "expose" => attrs.is_exposed = true,
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

        // Expect: Type
        let type_name = match tokens.next() {
            Some(TokenTree::Ident(id)) => id.to_string(),
            _ => panic!("expected type after `:`"),
        };

        // Check for [attrs]
        let mut attrs = ActorAttrs::default();
        if let Some(TokenTree::Group(g)) = tokens.peek() {
            if g.delimiter() == Delimiter::Bracket {
                let inner = g.stream().to_string();
                attrs = parse_attrs(&inner);
                tokens.next(); // consume the bracket group
            }
        }

        actors.push((name, type_name, attrs.is_main, attrs.is_exposed));

        // Skip comma if present
        if let Some(TokenTree::Punct(p)) = tokens.peek() {
            if p.as_char() == ',' {
                tokens.next();
            }
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
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Self>>::Data,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Self>>::Control,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Self>>::Management,
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
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Data,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Control,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Management,
                >,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate exposed() impl - clones exposed handles from directory
    let exposed_clone: String = exposed_actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}: self.directory.{name}.clone(),"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate scheduler fields for Troupe struct
    let scheduler_fields: String = actors
        .iter()
        .map(|(name, ty, _, _)| {
            format!(
                "{name}_scheduler: ::actor_scheduler::ActorScheduler<
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Data,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Control,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Management,
                >,"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate handle/scheduler creation in new()
    let create_actors: String = actors
        .iter()
        .map(|(name, ty, _, _)| {
            format!(
                "let ({name}_h, {name}_s) = ::actor_scheduler::create_actor::<
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Data,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Control,
                    <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Management,
                >(1024, None);"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate directory construction
    let dir_init: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}: {name}_h,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate scheduler init for Troupe struct
    let scheduler_init: String = actors
        .iter()
        .map(|(name, _, _, _)| format!("{name}_scheduler: {name}_s,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate spawns for non-main actors in play()
    let spawns: String = actors
        .iter()
        .filter(|(_, _, is_main, _)| !is_main)
        .map(|(name, ty, _, _)| {
            format!(
                r#"
                let mut {name}_s = self.{name}_scheduler;
                s.spawn(move || {{
                    let mut actor = <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::new(dir);
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
        let mut {main_name}_s = self.{main_name}_scheduler;
        let mut actor = <{main_ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::new(dir);
        {main_name}_s.run(&mut actor);
        "#
    );

    // Generate shutdown_all method
    let shutdown_impl = actors
        .iter()
        .map(|(name, _, _, _)| {
            format!("let _ = self.{name}.send(::actor_scheduler::Message::Control(Default::default()));")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let shutdown_bounds = actors
        .iter()
        .map(|(_, ty, _, _)| {
            format!("<{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Control: Default")
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"
        /// Directory containing handles to all actors in this troupe.
        pub struct Directory {{
            {dir_fields}
        }}

        impl Directory {{
            /// Send shutdown signal to all actors.
            /// Note: Each actor's Control type must implement Default.
            pub fn shutdown_all(&self)
            where
                {shutdown_bounds}
            {{
                {shutdown_impl}
            }}
        }}

        /// Handles exposed to parent troupes.
        pub struct ExposedHandles {{
            {exposed_fields}
        }}

        /// Troupe manages actor group lifecycle.
        ///
        /// Use `new()` to create, `exposed()` to get handles for parent,
        /// then `play()` to run actors in scoped threads.
        pub struct Troupe {{
            directory: Directory,
            {scheduler_fields}
        }}

        impl Troupe {{
            /// Create a new troupe. Builds directory and schedulers, but doesn't spawn threads.
            ///
            /// This is phase 1 of two-phase initialization:
            /// 1. `new()` - create channels, parent can grab exposed handles
            /// 2. `play()` - spawn threads, run actors
            pub fn new() -> Self {{
                {create_actors}

                let directory = Directory {{
                    {dir_init}
                }};

                Self {{
                    directory,
                    {scheduler_init}
                }}
            }}

            /// Get handles to exposed actors.
            ///
            /// Call this after `new()` but before `play()` to give parent
            /// troupe access to child actors.
            pub fn exposed(&self) -> ExposedHandles {{
                ExposedHandles {{
                    {exposed_clone}
                }}
            }}

            /// Get a reference to the directory.
            pub fn directory(&self) -> &Directory {{
                &self.directory
            }}

            /// Run the troupe. Spawns threads for non-main actors,
            /// runs main actor on calling thread. Blocks until main actor exits.
            ///
            /// This is phase 2 of two-phase initialization.
            pub fn play(self) -> ::std::result::Result<(), ::std::boxed::Box<dyn ::std::error::Error + Send + Sync>> {{
                let dir = self.directory;
                ::std::thread::scope(|s| {{
                    let dir = &dir;

                    {spawns}

                    {main_run}

                    Ok(())
                }})
            }}
        }}

        /// Convenience function: creates troupe and runs it.
        ///
        /// Equivalent to `Troupe::new().play()`.
        pub fn run() -> ::std::result::Result<(), ::std::boxed::Box<dyn ::std::error::Error + Send + Sync>> {{
            Troupe::new().play()
        }}
        "#,
    )
    .parse()
    .expect("failed to parse generated troupe code")
}
