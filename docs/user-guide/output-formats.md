# Output Formats

WhisperJAV generates subtitle files in standard formats compatible with all major media players.

---

## SRT (SubRip)

The default output format. Universally supported.

```
1
00:00:05,200 --> 00:00:08,100
こんにちは

2
00:00:10,500 --> 00:00:13,800
今日はいい天気ですね
```

**Supported by:** VLC, MPC-HC, mpv, PotPlayer, Windows Media Player, Plex, Kodi, and virtually every video player.

---

## VTT (WebVTT)

Web-native subtitle format for HTML5 `<video>` elements.

```
WEBVTT

00:00:05.200 --> 00:00:08.100
こんにちは

00:00:10.500 --> 00:00:13.800
今日はいい天気ですね
```

**Supported by:** All modern web browsers, YouTube, HTML5 video players. Also supported by VLC and mpv.

**Differences from SRT:** No sequence numbers, uses periods instead of commas for milliseconds, `WEBVTT` header required.

---

## Choosing a Format

| Format | Use when... |
|--------|-------------|
| **SRT** | Playing with desktop video players, uploading to most platforms |
| **VTT** | Embedding in web pages, HTML5 video players |
| **Both** | You need both — generates `.srt` and `.vtt` side by side |

### GUI

Go to **Advanced Options** → **Output Format** dropdown → select SRT, VTT, or Both.

### CLI

```bash
whisperjav video.mp4 --output-format vtt
whisperjav video.mp4 --output-format both
```

---

## Output File Location

By default, subtitle files are saved **next to the source video** with the same name:

```
C:\Videos\movie.mp4
C:\Videos\movie.srt      ← generated
C:\Videos\movie.vtt      ← if VTT or Both selected
```

To change this:

- **GUI:** Uncheck "Save next to source video" and browse to a custom folder
- **CLI:** Use `--output-dir /path/to/folder/`
