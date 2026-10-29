

import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from clipperx.schema import ClipCandidate
from clipperx.editing import render_clip, LayoutConfig

def test_dynamic_render():
    # Use a dummy video (assuming one exists or we fail)
    video_path = Path("o:/Escritorio/ClipperX/data/test_video.mp4")
    if not video_path.exists():
        print(f"Skipping test: {video_path} not found")
        # Use a dummy video
        video_path = Path("o:/Escritorio/ClipperX/data/dummy.mp4")
        if not video_path.exists():
            print(f"No dummy video found at {video_path}. Cannot proceed with test.")
            return

    print(f"Using video: {video_path}")
    output = Path("test_dynamic_output.mp4")
    
    # Create a dummy clip with TWO segments
    # 0-5s: SPLIT
    # 5-10s: FULL_CAM
    clip = ClipCandidate(
        start=0.0,
        end=10.0,
        text="Dummy transcript text",
        coherence=0.9,
        headline=None, # Disable title to avoid font issues
        layout_segments=[
            (0.0, 5.0, "SPLIT"),
            (5.0, 10.0, "FULL_CAM")
        ]
    )

    layout = LayoutConfig(
        face_crop=(0, 0, 200, 200),
        watermark_text="" # Disable watermark
    )
    
    print("Rendering dynamic clip...")
    try:
        render_clip(video_path, output, clip, layout=layout, title_text="Dynamic Test")
        print(f"Success! Output: {output}")
    except Exception as e:
        print(f"Error rendering: {e}")
        with open("ffmpeg_error.log", "w") as f:
            if hasattr(e, 'cmd'):
                f.write(f"Command: {e.cmd}\n\n")
            if hasattr(e, 'stderr'):
                err_text = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                f.write(err_text)
                print("FFmpeg stderr written to ffmpeg_error.log")
            else:
                f.write(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_render()
