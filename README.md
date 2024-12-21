# WhisperJAV

**A faster, easier way to create subtitles for your favorite JAV videos.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_v0_7b.ipynb)  
**Latest version:** `0.7b`

---

## Support This Project

If you’ve found WhisperJAV helpful, please consider supporting further development:

[![Buy Me a Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://buymeacoffee.com/meizhong)

Your support helps me dedicate more time to improvements and acquire the GPU and colab resources that I need to keep WhisperJAV development.

---

## Key Notes

WhisperJAV uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) to achieve roughly 2x the speed of the original Whisper, along with additional post-processing to remove hallucinations and repetition.


- **Faster with WAV:** The script runs *much* faster using WAV audio format.  
- **Hallucination Removal:** Currently more robust for Japanese transcription than for English translation tasks.  
- **Goal:** Make it *fast* and *easy* for non-technical users.

---

## Getting Started

### 1. Extract MP3 (or WAV) Audio from Videos
You’ll need an MP3 (or WAV) file of your video. Popular tools:

- [Clever FFmpeg GUI](https://www.videohelp.com/software/clever-FFmpeg-GUI)  
- [VLC Media Player](https://youtu.be/sMy-T8RJAo0?si=AKg-WgDAAhtaBFkr)  
- [Audacity](https://www.audacityteam.org/)  
- Direct `ffmpeg` command-line

> **Tip:** VLC and Audacity are user-friendly options if you prefer a graphical interface.

### 2. Create a “WhisperJAV” Folder in Google Drive
1. Go to [Google Drive](https://drive.google.com/).  
2. Click **+ New** \> **Folder**, name it `WhisperJAV`.  
3. Drag and drop or upload your MP3/WAV files into this folder.

See these quick tutorials for more info:  
- [Organize your files in Google Drive](https://support.google.com/drive/answer/2375091?hl=en&co=GENIE.Platform%3DDesktop)  
- [Managing files on Google Drive (Video)](https://youtu.be/EKjnjySLTvM?si=SF8ww3z572FnO_cq)

### 3. Run WhisperJAV
Open our latest Colab notebook:

[**WhisperJAV v0.7b**](https://colab.research.google.com/github/meizhong986/WhisperJAV/blob/main/notebook/WhisperJAV_v0_7b.ipynb)

1. Once the notebook is open, go to **Runtime** \> **Run all**.  
2. When prompted, allow Google Colab to connect to your Google Drive (this gives it permission to read/write your WhisperJAV folder).  
3. If you change any options or get odd errors, select **Runtime** \> **Restart and run all** for a clean start.  
4. Watch out for Captcha-style checks—Google may ask if you’re still there.

### 4. Download Subtitles
- Final subtitles are saved to your `WhisperJAV` folder in Drive.  
- Colab will also automatically zip and download them when processing completes (depending on your settings).  
- Alternatively, you can manually download from Drive.

---

## Credits and Citation
WhisperJAV would not be possible without prior works and contributions from:  
- @Anon_entity  
- @phineas-pta  
- JAV communities on [ScanLover](https://www.scanlover.com/) and [Akiba-Online](https://www.akiba-online.com/)


---

## Contact and Sponsorship
If you’re interested in sponsoring new features or need technical support, feel free to reach out:  
[**meizhong.986@gmail.com**](mailto:meizhong.986@gmail.com)

---

Thanks for using WhisperJAV, and happy subtitling!
