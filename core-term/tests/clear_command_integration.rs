use core_term::ansi::{AnsiParser, AnsiProcessor};
use core_term::io::pty::{NixPty, PtyConfig};
use core_term::term::{EmulatorInput, TerminalEmulator};
use std::io::{ErrorKind, Read};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn clear_command_emits_an_erase_sequence_when_parent_term_is_dumb() {
    // App launchers can supply TERM=dumb. CoreTerm must advertise its own
    // capabilities to the child instead of inheriting the launcher's terminal.
    std::env::set_var("TERM", "dumb");

    let config = PtyConfig {
        command_executable: "clear",
        args: &[],
        initial_cols: 80,
        initial_rows: 24,
    };
    let mut pty =
        NixPty::spawn_with_config(&config).expect("failed to spawn clear in CoreTerm PTY");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut output = Vec::new();
    let mut buffer = [0_u8; 256];

    loop {
        match pty.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => output.extend_from_slice(&buffer[..read]),
            Err(error) if error.kind() == ErrorKind::WouldBlock && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                panic!("clear did not exit within 5 seconds; output: {output:?}");
            }
            Err(error) => panic!("failed reading clear output: {error}; output: {output:?}"),
        }
    }

    assert!(
        output.starts_with(b"\x1b["),
        "clear must emit a terminal control sequence, got: {}",
        String::from_utf8_lossy(&output)
    );

    let mut parser = AnsiProcessor::new();
    let mut emulator = TerminalEmulator::new(80, 24);
    for command in parser.process_bytes(b"visible text") {
        drop(emulator.interpret_input(EmulatorInput::Ansi(command)));
    }
    assert_eq!(
        emulator
            .get_render_snapshot()
            .expect("missing seeded terminal snapshot")
            .lines[0]
            .cells[0]
            .display_char(),
        'v'
    );

    for command in parser.process_bytes(&output) {
        drop(emulator.interpret_input(EmulatorInput::Ansi(command)));
    }
    let snapshot = emulator
        .get_render_snapshot()
        .expect("clear did not produce a terminal snapshot");
    assert!(
        snapshot
            .lines
            .iter()
            .flat_map(|line| line.cells.iter())
            .all(|cell| cell.display_char() == ' '),
        "clear output reached the emulator but left visible cells behind"
    );
}
