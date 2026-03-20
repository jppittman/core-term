import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match the test attribute, any other attributes, fn definition, and body.
    # It attempts to capture the entire test block.
    # This might be tricky with nested braces, so we will use a more robust brace-matching approach.
    pass

# We will use string manipulation to find #[test] blocks, extract their bodies, check for assertions,
# and remove the entire function if necessary.

def remove_empty_tests(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []

    in_test = False
    test_start_line = -1
    brace_count = 0
    test_body = ""
    current_test_name = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        if '#[test]' in line:
            # Look ahead for 'fn '
            j = i + 1
            while j < len(lines) and 'fn ' not in lines[j] and '#' in lines[j]:
                j += 1

            if j < len(lines) and 'fn ' in lines[j]:
                match = re.search(r'fn\s+([a-zA-Z0-9_]+)\s*\(', lines[j])
                if match:
                    current_test_name = match.group(1)
                    in_test = True
                    test_start_line = i
                    brace_count = 0
                    test_body = ""

        if not in_test:
            new_lines.append(line)
            i += 1
            continue

        # We are inside a test or test header
        test_body += line + "\n"

        # Count braces
        brace_count += line.count('{')
        brace_count -= line.count('}')

        # If we have started seeing braces and now brace_count is 0, the test is over.
        # But we must be careful: if we haven't seen the first brace yet, brace_count is 0.
        # We can track if we've seen at least one brace.
        if '{' in line:
            seen_brace = True

        if in_test and '{' in test_body and brace_count == 0:
            # Test block complete

            # Check if it has assert
            # Strip comments to avoid matching asserts in comments
            body_no_comments = re.sub(r'//.*', '', test_body)
            body_no_comments = re.sub(r'/\*.*?\*/', '', body_no_comments, flags=re.DOTALL)

            has_assert = 'assert' in body_no_comments or 'expect(' in body_no_comments or 'unwrap()' in body_no_comments

            # Additional check for just assert!(true) or assert_eq!(true, true)
            # Remove whitespace for easier checking
            compact_body = body_no_comments.replace(' ', '').replace('\n', '').replace('\t', '')
            only_assert_true = False

            if 'assert!(true)' in compact_body or 'assert_eq!(true,true)' in compact_body:
                # How many asserts?
                if compact_body.count('assert') == 1:
                    only_assert_true = True

            if not has_assert or only_assert_true:
                # Remove this test
                print(f"Removing test {current_test_name} from {filepath}")
                # Do not append test_body to new_lines
            else:
                # Keep this test
                new_lines.extend(test_body.split('\n')[:-1]) # -1 because we added a trailing \n

            in_test = False

        i += 1

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))

for root, dirs, files in os.walk("."):
    if "target" in root: continue
    for file in files:
        if file.endswith(".rs"):
            remove_empty_tests(os.path.join(root, file))
