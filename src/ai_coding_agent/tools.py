import subprocess
from pathlib import Path

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"


@tool
def write_file(filename: str, content: str) -> str:
    """Create or overwrite a file inside the coding workspace."""

    WORKSPACE.mkdir(exist_ok=True)

    file_path = WORKSPACE / filename

    file_path.write_text(
        content,
        encoding="utf-8"
    )

    return f"File '{filename}' created successfully."


@tool
def read_file(filename: str) -> str:
    """Read a file from the coding workspace."""

    file_path = WORKSPACE / filename

    if not file_path.exists():
        return f"File '{filename}' does not exist."

    return file_path.read_text(
        encoding="utf-8"
    )


@tool
def list_files() -> str:
    """List all files inside the coding workspace."""

    WORKSPACE.mkdir(exist_ok=True)

    files = [
        str(file.relative_to(WORKSPACE))
        for file in WORKSPACE.rglob("*")
        if file.is_file()
    ]

    if not files:
        return "The workspace is empty."

    return "\n".join(files)


@tool
def execute(filename: str, arguments: str = "", **kwargs) -> str:
    """Execute a code file located in the workspace and return its output."""

    WORKSPACE.mkdir(exist_ok=True)
    file_path = WORKSPACE / filename

    if not file_path.exists():
        return f"File '{filename}' does not exist in workspace."

    cli_args = arguments or kwargs.get("args") or kwargs.get("v__args") or ""

    ext = file_path.suffix.lower()
    ws_dir = str(WORKSPACE.resolve())

    if ext == ".py":
        cmd = ["python3", file_path.name]
    elif ext in [".cpp", ".cc", ".cxx"]:
        bin_name = file_path.with_suffix(".out").name
        comp = subprocess.run(
            ["g++", "-std=c++17", file_path.name, "-o", bin_name],
            cwd=ws_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if comp.returncode != 0:
            return f"Compilation failed:\n{comp.stderr}"
        cmd = [f"./{bin_name}"]
    elif ext in [".js", ".mjs"]:
        cmd = ["node", file_path.name]
    elif ext == ".sh":
        cmd = ["bash", file_path.name]
    else:
        cmd = ["python3", file_path.name]

    if cli_args:
        cmd.extend(cli_args.split())

    try:
        proc = subprocess.run(
            cmd,
            cwd=ws_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\nStderr:\n{proc.stderr}"
        return output or "Executed successfully with no output."
    except subprocess.TimeoutExpired:
        return f"Execution of '{filename}' timed out after 10 seconds (the program may be waiting for input or in an infinite loop)."
    except Exception as e:
        return f"Error executing '{filename}': {str(e)}"


@tool
def delete(filename: str) -> str:
    """Delete a file from the workspace."""

    file_path = WORKSPACE / filename

    if not file_path.exists():
        return f"File '{filename}' does not exist in workspace."

    try:
        file_path.unlink()
        return f"File '{filename}' deleted successfully."
    except Exception as e:
        return f"Error deleting '{filename}': {str(e)}"


@tool
def shell(command: str) -> str:
    """Run a shell command inside the workspace directory."""

    WORKSPACE.mkdir(exist_ok=True)
    ws_dir = str(WORKSPACE.resolve())

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=ws_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = proc.stdout
        if proc.stderr:
            out += f"\nStderr:\n{proc.stderr}"
        return out or f"Command executed successfully (exit code {proc.returncode})."
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as e:
        return f"Shell execution error: {str(e)}"