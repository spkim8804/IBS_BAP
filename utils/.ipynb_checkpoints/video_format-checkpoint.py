import os
import subprocess
import re
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

class CheckVideoFormat(QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(str)
    
    def __init__(self, video_path, time = 1):
        super().__init__()
        self.video_path = video_path
        self.time = time
        
    def run(self):
        self.progress.emit("Checking video format..")
        output_video = self.extract_first_frames(self.video_path, self.time)
        frame_types = self.analyze_video(output_video)  # 추출된 파일 분석
        
        if "P" in frame_types:
            self.result.emit("P")
        elif "B" in frame_types:
            self.result.emit("B")
        elif "I" in frame_types:
            self.result.emit("I")

        os.remove(output_video)
        self.progress.emit("Checking video format completed!")
        
    def extract_first_frames(self, video_path, time, output_path="video_format_check.mp4"):
        ffmpeg_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffmpeg.exe")
        cmd = [
            ffmpeg_path,
            "-ss", "0",
            "-t", f"{time}",
            "-i", video_path,
            # "-c:v", "libx264",
            "-c:v", "copy",
            "-an",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_path
    
    def analyze_video(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        cmd = [
            ffprobe_path,
            "-i", video_path,
            "-show_frames",
            "-show_entries", "frame=pict_type",
            "-of", "csv"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        frames = [
            line.split(",")[-1] for line in result.stdout.splitlines()
            if line.startswith("frame") and line.split(",")[-1].strip() != ""
        ]
        return frames

class ConvertVideoToIframe(QThread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def get_bitrate(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        cmd = [
            ffprobe_path, 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=bit_rate", 
            "-of", "csv=p=0", self.video_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 결과값 정리
        if result.returncode == 0:  # 성공적으로 실행된 경우
            return result.stdout.strip()  # 결과값에서 공백 제거 후 반환
        else:
            return f"Error: {result.stderr}"  # 에러 메시지 반환
        
    def run(self):
        directory, filename = os.path.split(self.video_path)
        filename, ext = os.path.splitext(filename)
        output_path = f"{directory}/{filename}_iframe{ext}"
        
        bitrate = self.get_bitrate(self.video_path)
        print(f"Bitrate: {bitrate} bps")
        
        ffmpeg_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffmpeg.exe")
        cmd = [
            ffmpeg_path,
            "-i", self.video_path,
            "-g", "1", # I-frame encoding
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-an",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    