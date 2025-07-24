import os
import shutil
from pathlib import Path
import sys
import argparse
import re

# --- CONFIGURATION ---
# List of file names to exclude from the refactoring process.
EXCLUDED_FILES = [
    "main.py",
    "progress_display.py",
    "fast_pipeline.py",
    "faster_pipeline.py",
    "balanced_pipeline.py",
]

def find_files(root_dir: Path):
    """Scans the directory for .py files and categorizes them."""
    print(f"Scanning for Python files in: {root_dir.resolve()}\n")
    
    all_py_files = sorted(list(root_dir.rglob("*.py")))
    files_to_modify = []
    files_to_exclude = []

    script_path = Path(__file__).resolve()
    for file_path in all_py_files:
        if file_path.name in EXCLUDED_FILES or file_path.resolve() == script_path:
            files_to_exclude.append(file_path)
        else:
            files_to_modify.append(file_path)
    
    print(f"Found {len(all_py_files)} total Python files.")
    return files_to_modify, files_to_exclude

def run_dry_run(files_to_modify: list[Path], root_dir: Path):
    """Simulates the changes without writing any files."""
    print("--- [DRY RUN] Simulating changes. No files will be modified. ---")
    total_potential_changes = 0
    total_files_to_change = 0

    for file_path in files_to_modify:
        try:
            content = file_path.read_text(encoding='utf-8')
            # Use regex to find whole-word occurrences
            num_occurrences = len(re.findall(r'\blogger\.info\b', content))
            if num_occurrences > 0:
                print(f"  - Would change {num_occurrences} instance(s) in: {file_path.relative_to(root_dir)}")
                total_potential_changes += num_occurrences
                total_files_to_change += 1
        except Exception as e:
            print(f"  - Could not read {file_path.relative_to(root_dir)}: {e}", file=sys.stderr)
    
    print("\n--- [DRY RUN] Summary ---")
    print(f"Would change {total_potential_changes} lines across {total_files_to_change} files.")
    print("No changes have been made.")

def process_files(files_to_modify: list[Path], root_dir: Path):
    """Iterates through files and replaces logger.info with logger.debug."""
    total_files_changed = 0
    total_lines_changed = 0

    for file_path in files_to_modify:
        print(f"--- Processing file: {file_path.relative_to(root_dir)} ---")
        
        try:
            original_content = file_path.read_text(encoding='utf-8')
            
            # Use regex for a safer replacement
            modified_content, num_occurrences = re.subn(r'\blogger\.info\b', 'logger.debug', original_content)

            if num_occurrences == 0:
                print("  No 'logger.info' statements found. Skipping.")
                print(f"--- Finished file: {file_path.relative_to(root_dir)} ---\n")
                continue

            # Create backup path by appending .bak for clarity and robustness.
            backup_path = Path(f"{file_path}.bak")
            print(f"  [1/3] Found {num_occurrences} instance(s). Backing up original to: {backup_path.name}")
            shutil.copy2(file_path, backup_path)

            print(f"  [2/3] Saving {num_occurrences} modification(s)...")
            file_path.write_text(modified_content, encoding='utf-8')

            print("  [3/3] Verifying changes...")
            final_content = file_path.read_text(encoding='utf-8')
            
            if re.search(r'\blogger\.info\b', final_content):
                print(f"\n  !! CRITICAL: Verification failed for {file_path.name}. !!", file=sys.stderr)
                print(f"  !! Please review the file manually. A backup exists at {backup_path.name}. !!\n", file=sys.stderr)
            else:
                print("  Verification successful.")
                total_files_changed += 1
                total_lines_changed += num_occurrences

        except Exception as e:
            print(f"  An unexpected error occurred: {e}", file=sys.stderr)
            if 'backup_path' in locals() and backup_path.exists():
                print(f"  A backup of the original file should be available at {backup_path.name}")

        print(f"--- Finished file: {file_path.relative_to(root_dir)} ---\n")

    print("=" * 50)
    print("              Processing Complete")
    print("=" * 50)
    print(f"Summary: Changed {total_lines_changed} lines across {total_files_changed} files.")
    print("Original files backed up with a '.bak' extension.")
    print("=" * 50)

def reverse_changes(root_dir: Path):
    """Finds all .bak files and restores them."""
    print("--- Reverse Mode: Restoring from backups ---")
    # Search for any file ending in .bak inside the directory
    backup_files = sorted(list(root_dir.rglob("*.bak")))

    if not backup_files:
        print("No backup (.bak) files found to restore.")
        return

    print("The following files will be restored from their backups:")
    for bak_file in backup_files:
        # Use string manipulation for robustness (requires Python 3.9+ for removesuffix)
        original_name_str = str(bak_file).removesuffix('.bak')
        print(f"  - {Path(original_name_str).relative_to(root_dir)}")
    
    try:
        confirm = input("\nDo you want to restore these files? This will overwrite current files. (y/n): ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return
        
    if confirm.lower() != 'y':
        print("\nOperation cancelled. No files have been restored.")
        return
    
    print("\nRestoring files...")
    restored_count = 0
    for bak_file in backup_files:
        try:
            # Recreate the original path by removing the .bak suffix
            original_path = Path(str(bak_file).removesuffix('.bak'))
            shutil.move(bak_file, original_path)
            print(f"  - Restored {original_path.relative_to(root_dir)}")
            restored_count += 1
        except Exception as e:
            print(f"  - FAILED to restore {bak_file}: {e}", file=sys.stderr)

    print(f"\nRestore complete. {restored_count} files were restored.")


def main():
    """Entry point of the script."""
    script_dir = Path(__file__).parent.resolve()
    default_project_root = (script_dir / "..").resolve()
    
    parser = argparse.ArgumentParser(
        description="Globally refactor 'logger.info' to 'logger.debug' in a Python project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "path", nargs="?", default=str(default_project_root),
        help="The root directory of the project to scan.\nDefaults to the parent directory of the script."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan files and report what would be changed, but do not modify any files."
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Restore all '.bak' files, undoing the changes."
    )
    args = parser.parse_args()

    project_root = Path(args.path).resolve()

    if not project_root.is_dir():
        print(f"\nError: The specified path '{project_root}' is not a valid directory.", file=sys.stderr)
        return

    if args.reverse:
        reverse_changes(project_root)
        return

    files_to_modify, files_to_exclude = find_files(project_root)

    if not files_to_modify:
        print("No files found to modify. Exiting.")
        return

    print("\n--- Files to be MODIFIED ---")
    for file in files_to_modify:
        print(f"  - {file.relative_to(project_root)}")
    
    print("\n--- Files to be EXCLUDED ---")
    if files_to_exclude:
        print("  - " + "\n  - ".join(f.relative_to(project_root).as_posix() for f in files_to_exclude))
    else:
        print("  (None)")


    if args.dry_run:
        print("\n" + "="*50)
        run_dry_run(files_to_modify, project_root)
        print("="*50)
        return

    print("\n" + "="*50)
    print("The script will replace all instances of 'logger.info' with 'logger.debug'")
    print("in the files listed for MODIFICATION.")
    print("="*50)

    try:
        confirm = input("Do you want to proceed with these changes? (y/n): ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return
        
    if confirm.lower() == 'y':
        print("\nConfirmation received. Starting the modification process...\n")
        process_files(files_to_modify, project_root)
    else:
        print("\nOperation cancelled. No files have been changed.")

if __name__ == "__main__":
    main()
