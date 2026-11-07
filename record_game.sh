#!/usr/bin/env bash
# Record the game window to an MP4 video with ffmpeg (X11 only).
#
# Usage:
#   ./record_game.sh [-o out.mp4] [-r fps] [-- launch.py arguments...]
#
#   ./record_game.sh                                  # record a game that is already running
#   ./record_game.sh -- --food_types 3 --run          # start a game and record it
#   ./record_game.sh -o species.mp4 -r 60 -- --sexual --load_path new.ckpt --run
#
# Recording starts once the game window appears and stops when you press Ctrl-C, close the game,
# or the game exits. Ctrl-C stops only the recording: a game started by this script keeps running.
# ffmpeg records a fixed rectangle of the screen, so maximise or place the game window before
# recording starts and keep other windows off it.
set -euo pipefail

out="gameplay_$(date +%Y%m%d_%H%M%S).mp4"
fps=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        -r) fps="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option '$1' (see --help)" >&2; exit 1 ;;
    esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg is not installed" >&2; exit 1; }
command -v xwininfo >/dev/null || { echo "xwininfo is not installed (package x11-utils)" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is not set: this needs an X11 session" >&2; exit 1; }

cd "$(dirname "$0")"
game_pid=""
if [[ $# -gt 0 ]]; then   # start the game in its own session, so Ctrl-C here does not stop it
    python=python3
    [[ -x venv/bin/python ]] && python=venv/bin/python
    setsid "$python" ./launch.py "$@" &
    game_pid=$!
    echo "Started the game (pid $game_pid)"
fi

# (|| true: grep finding no window is not an error here, but set -e with pipefail would end the script)
find_window() { xwininfo -root -tree 2>/dev/null | grep '"CUDA-Evolution' | awk '{print $1}' | head -1 || true; }

echo "Waiting for the game window..."
win=""
for _ in $(seq 240); do   # up to 2 minutes: the first launch compiles the CUDA kernels
    win=$(find_window)
    [[ -n "$win" ]] && break
    if [[ -n "$game_pid" ]] && ! kill -0 "$game_pid" 2>/dev/null; then
        echo "The game exited before its window appeared" >&2; exit 1
    fi
    sleep 0.5
done
[[ -n "$win" ]] || { echo "No game window found" >&2; exit 1; }
sleep 2   # let the window manager finish placing and sizing the window

eval "$(xwininfo -id "$win" | awk '/Absolute upper-left X/{print "x="$4} /Absolute upper-left Y/{print "y="$4}
                                     /Width:/{print "w="$2} /Height:/{print "h="$2}')"
w=$((w / 2 * 2)); h=$((h / 2 * 2))   # H.264 needs even dimensions

if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    codec=(-c:v h264_nvenc -preset p5 -cq 23)      # NVIDIA hardware encoder: almost no CPU
else
    codec=(-c:v libx264 -preset veryfast -crf 23)
fi

echo "Recording ${w}x${h} at ${fps} fps to $out. Press Ctrl-C to stop."
ffmpeg -hide_banner -loglevel warning -y -nostdin -f x11grab -framerate "$fps" -video_size "${w}x${h}" \
       -i "${DISPLAY}+${x},${y}" "${codec[@]}" -pix_fmt yuv420p -movflags +faststart "$out" &
ffmpeg_pid=$!

# ffmpeg finishes the file on one SIGINT, but a second one makes it quit at once and leaves the
# file unplayable, so send it only once (Ctrl-C in this terminal also reaches ffmpeg directly)
stopping=""
stop() {
    [[ -n "$stopping" ]] && return
    stopping=1
    echo "Stopping: finishing the video file..."
    kill -INT "$ffmpeg_pid" 2>/dev/null || true
}
on_ctrl_c() { stopping=1; echo "Stopping: finishing the video file..."; }   # ffmpeg got the Ctrl-C itself
trap on_ctrl_c INT
trap stop TERM

while kill -0 "$ffmpeg_pid" 2>/dev/null; do
    if [[ -n "$game_pid" ]] && ! kill -0 "$game_pid" 2>/dev/null; then stop; fi
    if [[ -z "$(find_window)" ]]; then stop; fi
    sleep 1
done
wait "$ffmpeg_pid" 2>/dev/null || true
trap - INT TERM

if [[ -s "$out" ]]; then
    echo "Saved $out ($(du -h "$out" | cut -f1))"
else
    echo "Recording failed: $out is empty" >&2; exit 1
fi
