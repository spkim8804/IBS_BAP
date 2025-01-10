import cv2
import os
import subprocess

def export_frame_image(frame_data):
    frame, frame_id, image_path = frame_data
    cv2.imwrite(image_path, frame)

def get_frame_types(video_path):
    """ffprobe로 프레임 타입 가져오기"""
    ffprobe_path = os.path.join(os.getcwd(), "ffprobe.exe")  # ffprobe.exe 경로
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"ffprobe.exe not found in {os.getcwd()}")
    
    cmd = [
        ffprobe_path,
        "-i", video_path,
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pict_type",
        "-of", "csv"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error occurred: {result.stderr}")
        return []
    
    frames = [line.split(",")[-1] for line in result.stdout.splitlines() if line.startswith("frame")]
    return frames