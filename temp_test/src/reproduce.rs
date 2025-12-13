#[cfg(test)]
mod tests {
    use crate::ansi::{AnsiCommand, AnsiProcessor, AnsiParser};
    use crate::ansi::commands::EscCommand;

    fn process_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
        let mut processor = AnsiProcessor::new();
        processor.process_bytes(bytes)
    }

    #[test]
    fn test_esc_intermediate_strict_validation() {
        // Test case 1: '%' (0x25) as final char.
        // Current behavior: Accepted (because of the loose check)
        // Desired behavior: Rejected (Ignore)

        let bytes_percent = b"\x1B(%";
        let commands = process_bytes(bytes_percent);

        // This confirms current behavior matches what we think (it accepts it)
        // If we want to assert failure of validation (success of this test), we expect 'commands' to contain SelectCharacterSet.
        // But once we fix it, we expect it NOT to contain SelectCharacterSet.

        // So let's assert what we expect *after* fix. If I run this now, it should FAIL.
        // I want to see it fail first.

        let has_select = commands.iter().any(|c| matches!(c, AnsiCommand::Esc(EscCommand::SelectCharacterSet(_, _))));
        assert!(!has_select, "Validation failed: '%' should be rejected as final char");

        // Test case 2: '<' (0x3C) as final char.
        // Current behavior: Rejected (because is_ascii_alphanumeric check)
        // Desired behavior: Accepted (SelectCharacterSet)

        let bytes_less = b"\x1B(<";
        let commands_less = process_bytes(bytes_less);
        let has_select_less = commands_less.iter().any(|c| matches!(c, AnsiCommand::Esc(EscCommand::SelectCharacterSet(_, _))));
        assert!(has_select_less, "Validation failed: '<' should be accepted as final char");
    }
}
