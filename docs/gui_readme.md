# WhisperJAV GUI Implementation

## Overview
A clean, simple Tkinter-based GUI for WhisperJAV that follows your wireframe design exactly.

## Features Implemented

### ✅ Main Features
- **File Selection**: Select multiple media files for batch processing
- **Output Folder**: Defaults to input file location, can be changed
- **Speed Control**: Faster / Fast / Balanced modes
- **Granularity Control**: Aggressive / Balanced / Conservative sensitivity
- **Language Output**: Japanese or English direct translation
- **Advanced Settings**: View current configuration parameters
- **Console Output**: Real-time processing logs with auto-scroll
- **Start/Stop Button**: Control processing with visual feedback

### ✅ Design Elements
- Clean 4-column layout matching your wireframe
- User-friendly labels ("Quickie vs Less Mistakes", etc.)
- Gear icon (⚙) for settings
- Black console with colored output
- Responsive design with minimum window size
- Default values: Balanced/Balanced/Japanese

## How to Test

### 1. Quick Test (Without WhisperJAV)
```bash
# Run the test script to see the GUI in action
python test_gui.py
```
This opens the GUI in test mode with simulated processing.

### 2. Integration Test (With WhisperJAV)
```bash
# Run the actual GUI connected to WhisperJAV
python whisperjav_gui.py
```

### 3. Check GUI Elements
- Click "Select File(s)" - file dialog should open
- Select some media files - button updates with count
- Click "Output Folder" - folder dialog should open
- Try different radio button combinations
- Click gear icon "Advanced Settings" - popup should show current config
- Click "START PROCESSING" - should change to "STOP PROCESSING"

## File Structure
```
whisperjav_gui.py     # Main GUI implementation
test_gui.py          # Test script with mock processing
README.md            # This file
```

## Advanced Settings Dialog

The Advanced Settings dialog shows three tabs:
1. **VAD Options** - Voice Activity Detection parameters
2. **Transcribe Options** - Whisper transcription parameters  
3. **Decode Options** - Decoder parameters

These are read-only for now and show what will be used based on your Speed/Granularity selections.

## Console Output Colors
- **White**: General information
- **Yellow**: Warnings
- **Red**: Errors
- **Green**: Success messages

## Threading
The GUI uses threading to keep the interface responsive during processing:
- Main thread handles UI updates
- Processing thread runs WhisperJAV subprocess
- Queue system for thread-safe console updates

## Customization Points

### To Change Default Values
In `__init__`:
```python
self.speed_var = tk.StringVar(value="balanced")      # Change default mode
self.granularity_var = tk.StringVar(value="balanced") # Change default sensitivity  
self.language_var = tk.StringVar(value="japanese")    # Change default language
```

### To Modify Console Appearance
In `create_console_section`:
```python
self.console_text = tk.Text(console_container, wrap='word', 
                           bg='black', fg='white',      # Change colors
                           font=('Consolas', 9))        # Change font
```

### To Add More Settings
The `get_vad_settings`, `get_transcribe_settings`, and `get_decode_settings` methods can be expanded to show more parameters.

## Building EXE

To create a Windows executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller --onefile --windowed --name WhisperJAV whisperjav_gui.py

# The exe will be in dist/WhisperJAV.exe
```

## Future Enhancements (Mentioned in Design)
- Save/Load custom configurations
- Show selected files list
- Progress bar (currently WhisperJAV's own progress is shown)
- More advanced settings controls

## Troubleshooting

### GUI Freezes
- Check that WhisperJAV is properly installed
- Verify Python path is correct
- Look for errors in console output

### No Output
- Ensure files are selected
- Check output folder permissions
- Verify WhisperJAV works from command line first

### Advanced Settings Empty
- The settings are currently mock data
- Will need to integrate with actual WhisperJAV config system

## Notes
- Output folder defaults to input file location as requested
- Console auto-scrolls to show latest messages
- All controls are disabled during processing
- Stop button terminates the subprocess cleanly