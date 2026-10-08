# python script to generate the structure of the project
from pathlib import Path
import ast

# ============================================================
# Configuration
# ============================================================

# Directories that should be shown but NOT opened/expanded
COLLAPSED_DIRS = {
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    "node_modules",
    ".idea",
    ".vscode",
}

# Directories that should be completely hidden
HIDDEN_DIRS = {
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".coverage",
}

# Files that should be completely hidden
HIDDEN_FILES = {
    ".DS_Store",
    "Thumbs.db",
}

# File extensions that should be hidden
HIDDEN_EXTENSIONS = {
    ".pyc",
    ".pyo",
}


# ============================================================
# Filtering
# ============================================================


def should_hide(path: Path) -> bool:
    """Return True if this file/directory should not appear at all."""

    if path.is_dir() and path.name in HIDDEN_DIRS:
        return True

    if path.is_file() and path.name in HIDDEN_FILES:
        return True

    if path.is_file() and path.suffix in HIDDEN_EXTENSIONS:
        return True

    return False


def should_collapse(path: Path) -> bool:
    """Return True if a directory should be shown but not expanded."""
    return path.is_dir() and path.name in COLLAPSED_DIRS


# ============================================================
# Directory Tree
# ============================================================


def generate_tree(directory: Path, prefix: str = "") -> list[str]:
    """Generate a clean project tree."""

    lines = []

    try:
        entries = [entry for entry in directory.iterdir() if not should_hide(entry)]

        # Directories first, then files
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

    except PermissionError:
        return lines

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")

        # Show directory but don't open it
        if should_collapse(entry):
            lines.append(
                f"{prefix}{'    ' if is_last else '│   '}" f"└── [contents excluded]"
            )
            continue

        # Recursively expand normal directories
        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")

            lines.extend(generate_tree(entry, new_prefix))

    return lines


# ============================================================
# Python File Analysis
# ============================================================


def analyze_python_file(file_path: Path) -> dict:
    """Extract useful architecture information from a Python file."""

    result = {
        "classes": [],
        "functions": [],
        "imports": [],
    }

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")

        tree = ast.parse(source)

    except (SyntaxError, OSError):
        return result

    for node in ast.walk(tree):

        if isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)

    # Remove duplicates while preserving order
    result["classes"] = list(dict.fromkeys(result["classes"]))
    result["functions"] = list(dict.fromkeys(result["functions"]))
    result["imports"] = list(dict.fromkeys(result["imports"]))

    return result


def find_python_files(directory: Path) -> list[Path]:
    """Find Python files while respecting ignored directories."""

    python_files = []

    try:
        for path in directory.iterdir():

            if should_hide(path):
                continue

            if should_collapse(path):
                # Do not inspect venv, __pycache__, etc.
                continue

            if path.is_file() and path.suffix == ".py":
                if path.name not in HIDDEN_FILES:
                    python_files.append(path)

            elif path.is_dir():
                python_files.extend(find_python_files(path))

    except PermissionError:
        pass

    return python_files


# ============================================================
# Markdown Generation
# ============================================================


def generate_markdown(project_root: Path) -> str:

    tree = generate_tree(project_root)

    python_files = find_python_files(project_root)

    markdown = []

    markdown.append("# Project Architecture")
    markdown.append("")
    markdown.append(
        "Automatically generated project structure and Python "
        "architecture information."
    )
    markdown.append("")

    # --------------------------------------------------------
    # Directory Tree
    # --------------------------------------------------------

    markdown.append("## Directory Tree")
    markdown.append("")
    markdown.append("```text")
    markdown.append(f"{project_root.name}/")

    if tree:
        markdown.extend(tree)

    markdown.append("```")
    markdown.append("")

    # --------------------------------------------------------
    # Python Architecture
    # --------------------------------------------------------

    markdown.append("## Python Architecture")
    markdown.append("")

    if not python_files:
        markdown.append("No Python files found.")
        markdown.append("")
    else:

        for file_path in sorted(python_files):

            relative_path = file_path.relative_to(project_root)

            info = analyze_python_file(file_path)

            markdown.append(f"### `{relative_path}`")
            markdown.append("")

            if info["classes"]:
                markdown.append("**Classes:**")
                for class_name in info["classes"]:
                    markdown.append(f"- `{class_name}`")
                markdown.append("")

            if info["functions"]:
                markdown.append("**Functions:**")
                for function_name in info["functions"]:
                    markdown.append(f"- `{function_name}()`")
                markdown.append("")

            if info["imports"]:
                markdown.append("**Imports:**")
                for import_name in info["imports"]:
                    markdown.append(f"- `{import_name}`")
                markdown.append("")

            if not any(info.values()):
                markdown.append("No classes, functions, or imports detected.")
                markdown.append("")

    # --------------------------------------------------------
    # Excluded / Collapsed Directories
    # --------------------------------------------------------

    markdown.append("## Collapsed Directories")
    markdown.append("")

    markdown.append(
        "The following directories are intentionally shown in the "
        "tree but their contents are not expanded:"
    )
    markdown.append("")

    for directory in sorted(COLLAPSED_DIRS):
        markdown.append(f"- `{directory}/`")

    markdown.append("")

    markdown.append("## Hidden Directories")
    markdown.append("")

    markdown.append(
        "The following directories are completely excluded from "
        "the generated architecture:"
    )
    markdown.append("")

    for directory in sorted(HIDDEN_DIRS):
        markdown.append(f"- `{directory}/`")

    markdown.append("")

    return "\n".join(markdown)


# ============================================================
# Main
# ============================================================


def main():

    # The directory containing this script is the project root
    project_root = Path(__file__).resolve().parent.parent

    output_file = project_root / "PROJECT_STRUCTURE.md"

    content = generate_markdown(project_root)

    output_file.write_text(content, encoding="utf-8")

    print("Project architecture generated successfully.")
    print(f"Project root : {project_root}")
    print(f"Output file  : {output_file}")


if __name__ == "__main__":
    main()
