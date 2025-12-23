//! Proc macros for actor-scheduler troupe system.
//!
//! This crate provides two macros:
//! - `#[actor_impl]` - Transforms an impl block into an Actor trait impl
//! - `troupe!` - Generates a Directory struct and run function for actor groups

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

/// Generates a Directory struct and run function for an actor group.
///
/// # Syntax
///
/// ```ignore
/// troupe! {
///     actor_name: ActorType,
///     actor_name: ActorType [main],  // exactly one must be marked [main]
/// }
/// ```
///
/// # Example
///
/// ```ignore
/// troupe! {
///     engine: EngineActor,
///     vsync: VsyncActor,
///     display: DisplayActor [main],
/// }
/// ```
///
/// This generates:
/// - `pub struct Directory { ... }` with typed handle fields
/// - `pub fn run() -> Result<(), Box<dyn Error>>` using scoped threads
#[proc_macro]
pub fn troupe(input: TokenStream) -> TokenStream {
    // Parse: name: Type, name: Type [main], ...
    let mut actors: Vec<(String, String, bool)> = Vec::new(); // (name, type, is_main)
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

        // Check for [main]
        let mut is_main = false;
        if let Some(TokenTree::Group(g)) = tokens.peek() {
            if g.delimiter() == Delimiter::Bracket {
                let inner = g.stream().to_string();
                if inner.trim() == "main" {
                    is_main = true;
                    tokens.next(); // consume the [main]
                }
            }
        }

        actors.push((name, type_name, is_main));

        // Skip comma if present
        if let Some(TokenTree::Punct(p)) = tokens.peek() {
            if p.as_char() == ',' {
                tokens.next();
            }
        }
    }

    // Validate exactly one main
    let main_count = actors.iter().filter(|(_, _, m)| *m).count();
    if main_count != 1 {
        panic!(
            "exactly one actor must be marked [main], found {}",
            main_count
        );
    }

    // Generate Directory fields
    let dir_fields: String = actors
        .iter()
        .map(|(name, ty, _)| {
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

    // Generate handle/scheduler creation
    let create_actors: String = actors
        .iter()
        .map(|(name, ty, _)| {
            format!(
                "let ({name}_h, mut {name}_s) = ::actor_scheduler::create_actor::<
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
        .map(|(name, _, _)| format!("{name}: {name}_h,"))
        .collect::<Vec<_>>()
        .join("\n");

    // Generate spawns for non-main actors
    let spawns: String = actors
        .iter()
        .filter(|(_, _, is_main)| !is_main)
        .map(|(name, ty, _)| {
            format!(
                r#"
                s.spawn(|| {{
                    let mut actor = <{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::new(dir);
                    {name}_s.run(&mut actor);
                }});
                "#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Generate main actor run
    let (main_name, main_ty, _) = actors.iter().find(|(_, _, m)| *m).unwrap();
    let main_run = format!(
        r#"
        let mut actor = <{main_ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::new(dir);
        {main_name}_s.run(&mut actor);
        "#
    );

    // Generate shutdown_all method
    let shutdown_impl = actors
        .iter()
        .map(|(name, _, _)| {
            format!("let _ = self.{name}.send(::actor_scheduler::Message::Control(Default::default()));")
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"
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

        pub fn run() -> ::std::result::Result<(), ::std::boxed::Box<dyn ::std::error::Error + Send + Sync>> {{
            ::std::thread::scope(|s| {{
                {create_actors}

                let dir = Directory {{
                    {dir_init}
                }};
                let dir = &dir;

                {spawns}

                {main_run}

                Ok(())
            }})
        }}
        "#,
        shutdown_bounds = actors
            .iter()
            .map(|(_, ty, _)| {
                format!("<{ty} as ::actor_scheduler::TroupeActor<'_, Directory>>::Control: Default")
            })
            .collect::<Vec<_>>()
            .join(",\n")
    )
    .parse()
    .expect("failed to parse generated troupe code")
}
