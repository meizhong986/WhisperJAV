"""
WhisperJAV GUI Visual Tester
Test the UI without needing the actual WhisperJAV backend
"""

import os
import sys
import time
import random
import threading
from pathlib import Path

# Mock the REPO_ROOT for testing
sys.path.insert(0, str(Path(__file__).parent))

# Import the modernized GUI
from whisperjav_modern_gui import ModernWhisperGUI, HAS_CTK

class MockWhisperGUI(ModernWhisperGUI):
    """Mock version for visual testing without backend"""
    
    def __init__(self):
        super().__init__()
        self.title("WhisperJAV - Visual Test Mode")
        
        # Auto-populate with sample data
        self.populate_sample_data()
        
        # Add test controls
        self.add_test_controls()
    
    def populate_sample_data(self):
        """Add sample files for visual testing"""
        sample_files = [
            "/home/user/videos/sample_video_001.mp4",
            "/home/user/videos/sample_video_002.mp4",
            "/home/user/videos/japanese_drama_ep01.mkv",
            "/home/user/videos/anime_episode_12.mp4",
            "/home/user/audio/podcast_interview.mp3",
        ]
        
        for file in sample_files:
            self.file_listbox.insert('end', file)
    
    def add_test_controls(self):
        """Add special test controls"""
        if HAS_CTK:
            import customtkinter as ctk
            
            # Add test panel at the top
            test_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="yellow")
            test_frame.pack(fill="x", before=self.winfo_children()[0])
            
            ctk.CTkLabel(test_frame, text="🧪 TEST MODE", 
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="black").pack(side="left", padx=10, pady=5)
            
            # Theme switcher
            ctk.CTkButton(test_frame, text="Toggle Theme", width=100,
                         command=self.toggle_theme,
                         fg_color="orange", text_color="black").pack(side="left", padx=5)
            
            # Simulate progress
            ctk.CTkButton(test_frame, text="Simulate Progress", width=120,
                         command=self.simulate_progress,
                         fg_color="green", text_color="white").pack(side="left", padx=5)
            
            # Generate logs
            ctk.CTkButton(test_frame, text="Generate Logs", width=100,
                         command=self.generate_sample_logs,
                         fg_color="blue", text_color="white").pack(side="left", padx=5)
            
            # Show all states
            ctk.CTkButton(test_frame, text="Show All States", width=100,
                         command=self.demonstrate_states,
                         fg_color="purple", text_color="white").pack(side="left", padx=5)
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if HAS_CTK:
            import customtkinter as ctk
            current = ctk.get_appearance_mode()
            new_mode = "dark" if current == "Light" else "light"
            ctk.set_appearance_mode(new_mode)
            self.log_to_console(f"→ Switched to {new_mode} mode\n")
    
    def simulate_progress(self):
        """Simulate a processing operation"""
        def sim():
            self.log_to_console("\n" + "="*50 + "\n")
            self.log_to_console("Starting simulated processing...\n")
            self.set_processing_state(True)
            
            if HAS_CTK:
                # Simulate determinate progress
                steps = [
                    (0.1, "Initializing transcription engine..."),
                    (0.2, "Loading model: large-v3..."),
                    (0.3, "Processing: sample_video_001.mp4"),
                    (0.5, "Processing: sample_video_002.mp4"),
                    (0.7, "Processing: japanese_drama_ep01.mkv"),
                    (0.85, "Applying post-processing..."),
                    (0.95, "Generating output files..."),
                    (1.0, "✓ Complete!")
                ]
                
                for progress, message in steps:
                    self.after(0, self.progress.set, progress)
                    self.after(0, self.status_label.configure, text=message)
                    self.after(0, self.log_to_console, f"  → {message}\n")
                    time.sleep(1)
            else:
                # Just show messages for ttk version
                messages = [
                    "Initializing...",
                    "Loading model...",
                    "Processing files...",
                    "Generating output...",
                    "Complete!"
                ]
                for msg in messages:
                    self.after(0, self.log_to_console, f"  → {msg}\n")
                    time.sleep(1)
            
            self.after(0, self.set_processing_state, False)
            self.after(0, self.log_to_console, "Processing complete!\n" + "="*50 + "\n\n")
        
        thread = threading.Thread(target=sim, daemon=True)
        thread.start()
    
    def generate_sample_logs(self):
        """Generate sample console output"""
        sample_logs = [
            "WhisperJAV v2.0.0 - Initialized",
            "Using CUDA device: NVIDIA GeForce RTX 3080",
            "Model: whisper-large-v3 (1550M parameters)",
            "="*60,
            "Processing queue:",
            "  [1/5] sample_video_001.mp4 (23:45)",
            "  [2/5] sample_video_002.mp4 (45:12)",
            "  [3/5] japanese_drama_ep01.mkv (58:30)",
            "  [4/5] anime_episode_12.mp4 (24:00)",
            "  [5/5] podcast_interview.mp3 (1:15:00)",
            "="*60,
            "Starting transcription with settings:",
            "  - Mode: balanced",
            "  - Sensitivity: balanced",
            "  - Language: japanese",
            "  - Async: enabled (4 workers)",
            "",
            "🎬 Processing sample_video_001.mp4...",
            "  ├─ Extracting audio... done (2.3s)",
            "  ├─ Running VAD... done (1.1s)",
            "  ├─ Transcribing segments... ",
            "  │  ├─ Segment 1/15 [00:00-00:30]",
            "  │  ├─ Segment 2/15 [00:30-01:00]",
            "  │  └─ ...",
            "  ├─ Post-processing... done",
            "  └─ ✓ Saved to: output/sample_video_001.srt",
            "",
            "📊 Statistics:",
            "  - Total duration: 2:26:27",
            "  - Processing time: 18:35",
            "  - Speed: 7.9x realtime",
            "  - Words transcribed: 15,234",
            "",
        ]
        
        for line in sample_logs:
            self.log_to_console(line + "\n")
            self.update()
            time.sleep(0.05)  # Simulate typing effect
    
    def demonstrate_states(self):
        """Show different UI states quickly"""
        states = [
            ("Ready", "normal", "Ready to process"),
            ("Processing", "processing", "Processing 3 of 5 files..."),
            ("Warning", "warning", "⚠ Low memory warning"),
            ("Error", "error", "❌ Model failed to load"),
            ("Success", "success", "✓ All files processed"),
        ]
        
        def cycle():
            for name, state, message in states:
                self.after(0, self.log_to_console, f"\n→ Showing {name} state: {message}\n")
                
                if HAS_CTK:
                    # Change status label
                    self.after(0, self.status_label.configure, text=message)
                    
                    # Change progress bar color based on state
                    if state == "error":
                        self.after(0, self.progress.configure, progress_color="red")
                    elif state == "warning":
                        self.after(0, self.progress.configure, progress_color="orange")
                    elif state == "success":
                        self.after(0, self.progress.configure, progress_color="green")
                    else:
                        self.after(0, self.progress.configure, progress_color=["#3B8ED0", "#1F6AA5"])
                
                time.sleep(2)
            
            # Reset
            if HAS_CTK:
                self.after(0, self.progress.configure, progress_color=["#3B8ED0", "#1F6AA5"])
            self.after(0, self.status_label.configure, text="Ready")
        
        thread = threading.Thread(target=cycle, daemon=True)
        thread.start()
    
    def build_command_args(self):
        """Override to prevent actual execution"""
        self.log_to_console("\n🧪 TEST MODE: Would execute with these arguments:\n")
        args = super().build_command_args()
        for i in range(0, len(args), 2):
            if i+1 < len(args):
                self.log_to_console(f"    {args[i]} {args[i+1]}\n")
            else:
                self.log_to_console(f"    {args[i]}\n")
        return args
    
    def start_processing(self):
        """Override to prevent actual processing"""
        if self.proc:
            return
        
        if not self.validate_inputs():
            # For testing, override validation
            self.log_to_console("\n🧪 TEST MODE: Validation bypassed for testing\n")
        
        # Show what would be executed
        self.build_command_args()
        
        # Simulate processing instead
        self.simulate_progress()


def main():
    """Run the visual tester"""
    print("\n" + "="*60)
    print("WHISPERJAV GUI VISUAL TESTER")
    print("="*60)
    
    if not HAS_CTK:
        print("\n⚠ CustomTkinter not installed!")
        print("The app will run in fallback mode with limited visuals.")
        print("\nTo see the full modern UI, install CustomTkinter:")
        print("  pip install customtkinter")
        print("\nPress Enter to continue in fallback mode...")
        input()
    else:
        print("\n✓ CustomTkinter detected - Full visual mode enabled!")
        print("\nTest Controls:")
        print("  • Toggle Theme - Switch between light/dark mode")
        print("  • Simulate Progress - See progress animation")
        print("  • Generate Logs - See sample console output")
        print("  • Show All States - Cycle through UI states")
    
    print("\nKeyboard Shortcuts:")
    print("  • Ctrl+O: Add files")
    print("  • Ctrl+S: Start (simulated)")
    print("  • F1: Help")
    print("  • Escape: Cancel")
    print("\n" + "="*60 + "\n")
    
    # Run the test GUI
    app = MockWhisperGUI()
    
    # Show initial help in console
    app.after(1000, app.generate_sample_logs)
    
    app.mainloop()


if __name__ == "__main__":
    main()