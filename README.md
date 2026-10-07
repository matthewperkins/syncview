# syncview

View Open Ephys recordings (EMG, GI slow waves, …) side by side with a behaviour video, frame-locked to
the camera trigger. Scroll and zoom smoothly from milliseconds to whole multi-hour sessions, play back at
0.1–300× speed, and set filters per channel. A Python API also renders synchronized clips (video on top,
data scrolling underneath) for talks and papers.

Runs on macOS, Linux and Windows. An NVIDIA GPU is optional. It's used for video decoding when available;
otherwise the CPU decoder (FFmpeg via PyAV) is used, which is frame-identical.

## Install

You need Python 3.11 or newer. The simplest route is [uv](https://docs.astral.sh/uv/), which installs
`syncview` as a command in its own environment:

```bash
# macOS: install uv once (or: curl -LsSf https://astral.sh/uv/install.sh | sh)
brew install uv

# then, from a git URL ...
uv tool install git+<repo-url>
# ... or from a downloaded/unzipped copy of this folder
uv tool install ./syncview
```

If uv then says its tool folder "is not on your PATH", run `uv tool update-shell` once and open a new
terminal.

To update later: `uv tool upgrade syncview` (git install) or `uv tool install --reinstall ./syncview`.

With plain pip in a virtual environment, `pip install <repo-url or folder>` works the same way.

Optional extras:

- **NVIDIA GPU decoding** (Linux/Windows only): `uv tool install "syncview[gpu] @ git+<repo-url>"`.
- **Clip export** (Python API only, not needed for the viewer) needs the `ffmpeg` program:
  `brew install ffmpeg` on macOS.

## Quick start

```bash
syncview --rec "/path/to/2026-05-21_09-42-03/Record Node 101/experiment2/recording1" \
         --video /path/to/BASLER_CAM_2026-05-21_09:52:51.mp4
```

- `--rec` is the Open Ephys **recording folder** (the one containing `structure.oebin`).
- `--video` is optional. You can also attach a video later with the **Video…** button.
- The first time a recording is opened, each channel is filtered once over the whole session in the
  background (see the progress bar). After that, any zoom level draws instantly. Expect roughly a
  minute for 16 channels of a 2-hour recording on a recent multi-core machine.

## What the data must look like

- **Open Ephys binary format** (`structure.oebin`, `continuous/…/continuous.dat`, `events/…/TTL/`).
  The default stream is `acquisition_board`; use `--stream` for another name. If it's wrong, the error
  lists the streams available.
- **One TTL pulse per video frame** on one digital line (default line 1, `--trigger-line N`).
  Frame *k* of the video is matched to the *k*-th rising edge on that line. Up to one extra trigger
  after the last frame is expected and ignored, e.g. one sent when stopping the camera.
- Frames are counted from the file itself, not from its timestamps, so containers with jittery or
  drifting timestamps still sync exactly.

When a video is attached, syncview checks the alignment and warns (dialog and terminal) about:
too few triggers for the video (wrong file pair or trigger line; it suggests a matching line if one
exists), several extra triggers, an extra trigger that looks like a lost video frame, missed or doubled
triggers in the middle, a video duration that disagrees with the trigger span, and dropped samples in
the recording. **Take these warnings seriously.** They mean data and video may be offset.

## Options

| option | default | |
|---|---|---|
| `--rec FOLDER` | (required) | Open Ephys recording folder |
| `--video FILE` | none | video recorded during this recording (one camera) |
| `--preset FILE.json` | built from the recording | channel layout and filters, see below |
| `--stream NAME` | `acquisition_board` | Open Ephys continuous stream |
| `--trigger-line N` | `1` | TTL line with one pulse per video frame |
| `--decoder auto\|gpu\|cpu` | `auto` | video decoding (auto = NVIDIA GPU if available) |
| `--cache FOLDER` | see [Cache](#cache) | where filtered traces are stored |

## Controls

Click a plot first so it has keyboard focus.

| action | mouse / trackpad | keys |
|---|---|---|
| pan in time | drag; sideways swipe; Shift+wheel | ← / → one video frame; Shift+← / → 10 % of the view |
| zoom (time base) | scroll up/down | PgUp / PgDn: one full view back/forward |
| scale one row's Y | Ctrl+wheel (**⌘+scroll on macOS**) | |
| reset a row's Y to auto | double-click the row | |
| jump anywhere | click/drag the overview strip at the bottom | Home / End: start / end |
| play / pause, speed | ▶ button, speed menu | Space; [ / ] slower / faster |

Mac laptop keyboards: **fn+↑ / fn+↓** = PgUp / PgDn, **fn+← / fn+→** = Home / End.

## Channel presets

With no `--preset`, every electrode channel of the recording is shown (ADC/AUX inputs skipped), labelled
by its channel name, using the EMG (`hilo`) display. The first 16 are visible; tick the others on in the
Channels panel.

In the **Channels** panel you can reorder rows, rename them, show or hide them, set a bipolar reference
channel, change the display mode and its band edges, notch filter and Y range, and **save the layout as
a preset**. Load it next time with `--preset my_layout.json` or the panel's Load button. A preset that
names channels this recording doesn't have still loads: those rows are skipped with a warning.

Display modes:

| mode | what it shows | default filter |
|---|---|---|
| `hilo` | raw-rate EMG, drawn as per-pixel min/max so no spike is ever hidden | band-pass 150–4000 Hz |
| `envelope` | rectified, smoothed EMG | 150–4000 Hz → 20 ms boxcar → 40 Hz low-pass |
| `slow` | GI slow waves / slow potentials | band-pass 0.03–100 Hz, plotted at 1 kHz |
| `bandpower` | slow-wave power over minutes (used for the overview strip) | 0.03–0.25 Hz, RMS over 30 s |

A preset is plain JSON:

```json
{
  "time_base": 60,
  "channels": [
    {"ch": "CH13", "label": "masseter1", "mode": "hilo"},
    {"ch": "CH5", "ref": "CH6", "label": "antrum 5-6", "mode": "slow", "band": [0.05, 50]}
  ],
  "overview": {"ch": "CH5", "mode": "bandpower", "label": "antrum slow-wave power"}
}
```

## Cache

Filtered traces are saved so that every later view is instant. They are stored in, in order of
preference:

1. `--cache FOLDER`
2. the `SYNCVIEW_CACHE` environment variable
3. the per-user cache folder: `~/Library/Caches/syncview` (macOS), `~/.cache/syncview` (Linux),
   `%LOCALAPPDATA%\syncview` (Windows)

The viewer prints the folder it uses at startup. **It can get large.** An EMG (`hilo`) channel takes
about 340 MB per 2 hours of 10 kHz data, so 16 channels need about 5.5 GB per recording. Slow and
envelope traces are much smaller. Everything in it can be deleted at any time; it is rebuilt when
needed. To keep it on an external or faster drive, add e.g.
`export SYNCVIEW_CACHE="/Volumes/Data/syncview_cache"` to your shell profile (`~/.zshrc` on macOS).

The cache also holds a small index per video (`video_index/`, a few MB each) so videos open instantly
after the first time.

## Rendering clips (Python)

```python
from syncview.core import OERecording, make_sync_video

rec = OERecording("…/experiment2/recording1")
channels = [dict(ch="CH13", label="masseter", mode="hilo"), dict(ch="CH5", label="antrum", mode="slow")]
make_sync_video("clip.mp4", rec, ["…/BASLER_CAM_….mp4"], channels,
                start=1234.5, start_ref="ephys", duration=20, time_base=4.0)
```

`start_ref` is `"frame"` (video frame index), `"video"` (seconds as a video player shows it) or
`"ephys"` (seconds since the recording started). Needs `ffmpeg` on the PATH.

## Troubleshooting

- **"Fewer triggers than video frames"**: wrong recording/video pair, or the camera trigger is on
  another line (the message names a line whose count matches, if any). Try `--trigger-line`.
- **Video won't open / is slow to open the first time**: the first open of a large video reads its
  timestamps once (seconds to tens of seconds for multi-GB files); later opens are instant. `--decoder
  cpu` rules out GPU problems.
- **Scrolling stutters right after opening a recording**: the background cache build is still running;
  views are processed on demand until it finishes.
