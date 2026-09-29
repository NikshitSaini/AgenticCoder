import subprocess
from pathlib import Path

from ai_coding_agent.state import CodingState


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"


def check_output_match(actual: str, expected: str) -> bool:
    """Check if actual output matches or satisfies the expected output."""
    if not expected:
        return True
    actual_clean = actual.strip()
    expected_clean = expected.strip()
    if actual_clean == expected_clean:
        return True
    if expected_clean in actual_clean:
        return True
    # Whitespace-insensitive comparison
    if "".join(actual_clean.split()) == "".join(expected_clean.split()):
        return True
    return False



def tester(state: CodingState):

    print("\n--- TESTER ---")

    filename = state.get("filename", "")

    if not filename:
        return {
            "test_result": "FAIL: No filename was provided."
        }

    file_path = WORKSPACE / filename

    if not file_path.exists():
        return {
            "test_result": (
                f"FAIL: File '{filename}' does not exist in workspace."
            )
        }

    WORKSPACE.mkdir(exist_ok=True)
    ws_dir = str(WORKSPACE.resolve())

    # Always read the actual file content produced by write_file from disk
    try:
        actual_code = file_path.read_text(encoding="utf-8")
    except Exception:
        actual_code = ""

    try:
        ext = file_path.suffix.lower()
        language = state.get("language", "").lower()

        # Multi-language execution router
        if ext in [".cpp", ".cc", ".cxx"] or language == "cpp":
            binary_name = file_path.with_suffix(".out").name
            compile_proc = subprocess.run(
                ["g++", "-std=c++17", file_path.name, "-o", binary_name],
                cwd=ws_dir,
                capture_output=True,
                text=True,
                timeout=15
            )
            if compile_proc.returncode != 0:
                return {
                    "test_result": (
                        "FAIL\n"
                        f"Compilation Error:\n{compile_proc.stderr}"
                    ),
                    "code": actual_code,
                }
            run_cmd = [f"./{binary_name}"]
        elif ext in [".js", ".mjs"] or language in ["javascript", "js"]:
            run_cmd = ["node", file_path.name]
        elif ext in [".sh"] or language in ["bash", "sh"]:
            run_cmd = ["bash", file_path.name]
        else:
            run_cmd = ["python3", file_path.name]

        test_cases = state.get("test_cases", [])

        # Execute structured test cases if present
        if test_cases:
            results_summary = []
            all_passed = True

            for idx, tc in enumerate(test_cases, 1):
                input_str = tc.get("input", "")
                expected_str = tc.get("expected_output", "").strip()

                # Clean input for stdin: ignore if it's a CLI command invocation like 'python file.py'
                stdin_input = None
                if input_str and not input_str.strip().lower().startswith(("python", "node", "g++", "./", "bash")):
                    stdin_input = input_str.strip() + "\n"

                proc = subprocess.run(
                    run_cmd,
                    cwd=ws_dir,
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if proc.returncode != 0:
                    all_passed = False
                    results_summary.append(
                        f"Test Case {idx}: FAIL (Runtime Error)\n"
                        f"Input: {input_str}\n"
                        f"Error:\n{proc.stderr.strip()}"
                    )
                    continue

                actual_out = proc.stdout.strip()
                passed = check_output_match(actual_out, expected_str)

                if passed:
                    results_summary.append(
                        f"Test Case {idx}: PASS\n"
                        f"Input: {input_str}\n"
                        f"Expected: {expected_str}\n"
                        f"Actual: {actual_out}"
                    )
                else:
                    all_passed = False
                    results_summary.append(
                        f"Test Case {idx}: FAIL (Output Mismatch)\n"
                        f"Input: {input_str}\n"
                        f"Expected: {expected_str}\n"
                        f"Actual: {actual_out}"
                    )

            overall_status = "PASS" if all_passed else "FAIL"
            final_report = f"{overall_status}\n" + "\n---\n".join(results_summary)

            return {
                "test_result": final_report,
                "code": actual_code,
            }

        # Single run fallback if no test_cases are specified
        result = subprocess.run(
            run_cmd,
            cwd=ws_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Program crashed
        if result.returncode != 0:
            return {
                "test_result": (
                    "FAIL\n"
                    f"Runtime Error:\n{result.stderr}"
                ),
                "code": actual_code,
            }

        actual_output = result.stdout.strip()
        expected_output = state.get("expected_output", "").strip()

        # No expected output was provided
        if not expected_output:
            return {
                "test_result": (
                    "PASS\n"
                    f"Output:\n{actual_output}"
                ),
                "code": actual_code,
            }

        # Functional test comparison
        if check_output_match(actual_output, expected_output):
            return {
                "test_result": (
                    "PASS\n"
                    f"Expected: {expected_output}\n"
                    f"Actual: {actual_output}"
                ),
                "code": actual_code,
            }

        # Program ran but produced the wrong result
        return {
            "test_result": (
                "FAIL\n"
                f"Expected: {expected_output}\n"
                f"Actual: {actual_output}"
            ),
            "code": actual_code,
        }

    except subprocess.TimeoutExpired:
        return {
            "test_result": "FAIL: Program timed out.",
            "code": actual_code if 'actual_code' in locals() else "",
        }

    except Exception as e:
        return {
            "test_result": f"FAIL: {str(e)}",
            "code": actual_code if 'actual_code' in locals() else "",
        }
