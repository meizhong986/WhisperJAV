import os
import shutil
import re
from pathlib import Path

# --- CONFIGURATION: Define all the changes to be made ---
# This structure defines which files to target and what specific lines to change.
# For most files, it's a direct mapping of an old line to a new line.
# A special case is handled for 'hallucination_remover.py' to demonstrate
# a more complex, context-aware replacement.

CHANGES_TO_MAKE = [
    {
        "file": "audio_extraction.py",
        "replacements": [
            ('logger.info(f"Found FFmpeg at: {ffmpeg}")', 'logger.debug(f"Found FFmpeg at: {ffmpeg}")'),
            ('logger.info(f"Extracting audio from {input_file.name}")', 'logger.debug(f"Extracting audio from {input_file.name}")'),
            ('logger.info(f"Audio extracted successfully: {output_path.name} (duration: {duration:.1f}s)")', 'logger.debug(f"Audio extracted successfully: {output_path.name} (duration: {duration:.1f}s)")')
        ]
    },
    {
        "file": "whisper_pro_asr.py",
        "replacements": [
            ('logger.info("Loading Silero VAD model...")', 'logger.debug("Loading Silero VAD model...")'),
            ('logger.info(f"Loading Whisper model: {self.model_name}")', 'logger.debug(f"Loading Whisper model: {self.model_name}")')
        ]
    },
    {
        "file": "stable_ts_asr.py",
        "replacements": [
            ('logger.info(f"Loading Stable-TS model: {self.model_name}")', 'logger.debug(f"Loading Stable-TS model: {self.model_name}")'),
            ('logger.info(f"Transcribing: {audio_path.name}")', 'logger.debug(f"Transcribing: {audio_path.name}")'),
            ('logger.info(f"Saving SRT to: {output_srt_path}")', 'logger.debug(f"Saving SRT to: {output_srt_path}")')
        ]
    },
    {
        # Special case: Modify all logger.info calls within the __init__ method.
        "file": "hallucination_remover.py",
        "special_case": "init_method_log_to_debug"
    },
    {
        "file": "repetition_cleaner.py",
        "replacements": [
            ('logger.info(f"RepetitionCleaner initialized with {len(self.cleaning_patterns)} final cleaning patterns.")', 'logger.debug(f"RepetitionCleaner initialized with {len(self.cleaning_patterns)} final cleaning patterns.")')
        ]
    },
    {
        "file": "subtitle_sanitizer.py",
        "replacements": [
            ('logger.info("Initialized processors for rule-based cleaning.")', 'logger.debug("Initialized processors for rule-based cleaning.")'),
            ('logger.info(f"SubtitleSanitizer initialized with sensitivity: {self.config.sensitivity_mode}")', 'logger.debug(f"SubtitleSanitizer initialized with sensitivity: {self.config.sensitivity_mode}")'),
            ('logger.info(f"Saved sanitized SRT to: {output_path}")', 'logger.debug(f"Saved sanitized SRT to: {output_path}")'),
            ('logger.info(f"Saved artifacts to: {artifacts_path}")', 'logger.debug(f"Saved artifacts to: {artifacts_path}")')
        ]
    },
    {
        "file": "srt_stitching.py",
        "replacements": [
            ('logger.info(f"Combining {len(scene_srt_info)} scene SRT files")', 'logger.debug(f"Combining {len(scene_srt_info)} scene SRT files")'),
            ('logger.info(f"Successfully combined {len(all_subtitles)} subtitles into {output_path}")', 'logger.debug(f"Successfully combined {len(all_subtitles)} subtitles into {output_path}")')
        ]
    }
]

def get_indent_level(line: str) -> int:
    """Calculates the indentation level (number of leading spaces) of a line."""
    return len(line) - len(line.lstrip(' '))

def process_special_case_init_method(lines: list[str]) -> tuple[list[str], int]:
    """
    Handles the special case for hallucination_remover.py.
    Changes all 'logger.info' to 'logger.debug' only within the __init__ method.
    """
    new_lines = []
    changes_count = 0
    in_init_method = False
    init_indent_level = -1

    for line in lines:
        processed_line = line
        
        # Detect the start of the __init__ method
        if 'def __init__' in line:
            in_init_method = True
            init_indent_level = get_indent_level(line)
        elif in_init_method:
            # Detect the end of the __init__ method by a change in indentation
            current_indent = get_indent_level(line)
            if line.strip() and current_indent <= init_indent_level:
                in_init_method = False

        # If inside the method, perform the replacement
        if in_init_method and 'logger.info' in line:
            processed_line = line.replace('logger.info', 'logger.debug', 1)
            changes_count += 1
            print(f"    - Changing '{line.strip()}'\n      -> to '{processed_line.strip()}'")

        new_lines.append(processed_line)

    return new_lines, changes_count

def find_project_files(root_dir: Path, file_name: str) -> list[Path]:
    """Finds all occurrences of a file within the project directory."""
    found_files = list(root_dir.rglob(file_name))
    return found_files

def process_files(root_dir: Path):
    """
    Main function to iterate through the change definitions and process each file.
    """
    if not root_dir.is_dir():
        print(f"Error: Directory not found at '{root_dir}'")
        return

    print(f"Starting file processing in directory: {root_dir.resolve()}\n")

    total_files_changed = 0
    total_lines_changed = 0

    for change_info in CHANGES_TO_MAKE:
        file_name = change_info["file"]
        
        # Find the file(s) in the project structure
        target_files = find_project_files(root_dir, file_name)

        if not target_files:
            print(f"--- SKIPPING: Could not find '{file_name}' in '{root_dir}' ---\n")
            continue
        
        for file_path in target_files:
            print(f"--- Processing file: {file_path.relative_to(root_dir)} ---")

            try:
                # 1. Create a backup
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                print(f"  [1/4] Backing up original file to: {backup_path.name}")
                shutil.copy2(file_path, backup_path)

                # 2. Read the file content
                original_lines = file_path.read_text(encoding='utf-8').splitlines()
                new_lines = []
                lines_changed_in_file = 0
                
                print("  [2/4] Searching for lines to modify...")
                
                # 3. Modify the lines
                if "special_case" in change_info:
                    if change_info["special_case"] == "init_method_log_to_debug":
                        new_lines, lines_changed_in_file = process_special_case_init_method(original_lines)
                else:
                    # Standard line-by-line replacement
                    replacements_map = {old.strip(): new for old, new in change_info["replacements"]}
                    for line in original_lines:
                        stripped_line = line.strip()
                        if stripped_line in replacements_map:
                            new_line_content = replacements_map[stripped_line]
                            # Re-apply original indentation
                            indent = ' ' * get_indent_level(line)
                            new_line = indent + new_line_content
                            new_lines.append(new_line)
                            lines_changed_in_file += 1
                            print(f"    - Changing '{stripped_line}'\n      -> to '{new_line_content}'")
                        else:
                            new_lines.append(line)
                
                if lines_changed_in_file == 0:
                    print("  No applicable lines found to change in this file.")
                    # Clean up backup if no changes were made
                    backup_path.unlink()
                    print(f"  Removed unnecessary backup: {backup_path.name}")
                    print(f"--- Finished file: {file_path.relative_to(root_dir)} ---\n")
                    continue

                # 4. Save the modified file
                print(f"  [3/4] Saving {lines_changed_in_file} modification(s) to {file_path.name}...")
                # Join with os.linesep to respect OS-specific line endings
                file_path.write_text(os.linesep.join(new_lines) + os.linesep, encoding='utf-8')

                # 5. Verify the changes
                print("  [4/4] Verifying changes...")
                final_content = file_path.read_text(encoding='utf-8')
                verification_passed = True
                
                if "replacements" in change_info:
                    for old, new in change_info["replacements"]:
                        # Check that the old line is gone and the new one is present
                        if old in final_content:
                            print(f"  VERIFICATION FAILED: Old line still exists -> {old}")
                            verification_passed = False
                        if new not in final_content:
                            print(f"  VERIFICATION FAILED: New line not found -> {new}")
                            verification_passed = False

                if verification_passed:
                    print("  Verification successful. All changes confirmed.")
                    total_files_changed += 1
                    total_lines_changed += lines_changed_in_file
                else:
                    print(f"\n  !! CRITICAL: Verification failed for {file_path.name}. !!")
                    print(f"  !! Please review the file manually. A backup exists at {backup_path.name}. !!\n")

            except FileNotFoundError:
                print(f"  Error: File not found at the specified path: {file_path}")
            except Exception as e:
                print(f"  An unexpected error occurred: {e}")
                print(f"  A backup of the original file should be available at {backup_path.name}")

            print(f"--- Finished file: {file_path.relative_to(root_dir)} ---\n")

    print("===================================================")
    print("              Processing Complete")
    print("===================================================")
    print(f"Summary: Changed {total_lines_changed} lines across {total_files_changed} files.")
    print("Original files have been backed up with a '.bak' extension.")
    print("===================================================")


def main():
    """Entry point of the script."""
    # Assume the script is in a 'scripts' folder at the project root.
    # We navigate up one level to find the project root.
    script_dir = Path(__file__).parent
    default_project_root = (script_dir / "..").resolve()
    
    print("WhisperJAV Logging Level Refactoring Script")
    print("-------------------------------------------")
    print("This script will change specific 'logger.info' calls to 'logger.debug'.")
    print(f"It assumes the project root is: {default_project_root}")
    
    # Prompt the user to confirm the project root directory
    user_input = input(f"Press Enter to use this directory, or enter a new path: ")
    
    project_root = Path(user_input.strip()) if user_input.strip() else default_project_root

    if not project_root.is_dir():
        print(f"\nError: The specified path '{project_root}' is not a valid directory.")
        return

    process_files(project_root)


if __name__ == "__main__":
    main()
