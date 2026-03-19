# 输出格式

WhisperJAV 生成标准格式的字幕文件，兼容所有主流媒体播放器。

---

## SRT (SubRip)

默认输出格式。几乎所有播放器都支持。

```
1
00:00:05,200 --> 00:00:08,100
こんにちは

2
00:00:10,500 --> 00:00:13,800
今日はいい天気ですね
```

**支持的播放器：** VLC、MPC-HC、mpv、PotPlayer、Windows Media Player、Plex、Kodi，以及几乎所有视频播放器。

---

## VTT (WebVTT)

Web 原生字幕格式，用于 HTML5 `<video>` 元素。

```
WEBVTT

00:00:05.200 --> 00:00:08.100
こんにちは

00:00:10.500 --> 00:00:13.800
今日はいい天気ですね
```

**支持的播放器：** 所有现代 Web 浏览器、YouTube、HTML5 视频播放器。VLC 和 mpv 也支持。

**与 SRT 的区别：** 无序号、毫秒使用点号而非逗号、需要 `WEBVTT` 头部。

---

## 选择格式

| 格式 | 适用场景 |
|------|----------|
| **SRT** | 在桌面视频播放器中播放、上传到大多数平台 |
| **VTT** | 嵌入网页、HTML5 视频播放器 |
| **Both** | 两种都需要 — 同时生成 `.srt` 和 `.vtt` 文件 |

### GUI

进入 **Advanced Options** → **Output Format** 下拉菜单 → 选择 SRT、VTT 或 Both。

### CLI

```bash
whisperjav video.mp4 --output-format vtt
whisperjav video.mp4 --output-format both
```

---

## 输出文件位置

默认情况下，字幕文件保存在**源视频旁边**，使用相同的文件名：

```
C:\Videos\movie.mp4
C:\Videos\movie.srt      ← 生成的文件
C:\Videos\movie.vtt      ← 选择 VTT 或 Both 时生成
```

更改输出位置：

- **GUI：** 取消勾选 "Save next to source video" 并浏览选择自定义文件夹
- **CLI：** 使用 `--output-dir /path/to/folder/`
