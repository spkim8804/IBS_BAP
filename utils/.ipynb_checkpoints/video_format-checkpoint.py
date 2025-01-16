import sys
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
            # "ffmpeg",
            "-ss", "0",
            "-t", f"{time}",
            "-i", video_path,
            # "-c:v", "libx264",
            "-c:v", "copy",
            "-an",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return output_path
    
    def analyze_video(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        cmd = [
            ffprobe_path,
            # "ffprobe",
            "-i", video_path,
            "-show_frames",
            "-show_entries", "frame=pict_type",
            "-of", "csv"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        frames = [
            line.split(",")[-1] for line in result.stdout.splitlines()
            if line.startswith("frame") and line.split(",")[-1].strip() != ""
        ]
        return frames

class ConvertVideoToIframe(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    result = pyqtSignal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True

    def get_total_duration(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        command = [
            ffprobe_path,
            # "ffprobe",
            "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            raise RuntimeError(f"Error getting duration: {result.stderr}")

    def get_bitrate(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        command = [
            ffprobe_path, 
            # "ffprobe",
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            raise RuntimeError(f"Error getting bitrate: {result.stderr}")
 
    def run(self):
        directory, filename = os.path.split(self.video_path)
        filename, ext = os.path.splitext(filename)
        output_path = f"{directory}/{filename}_iframe{ext}"
        
        bitrate = self.get_bitrate(self.video_path)
        
        try:
            ffmpeg_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffmpeg.exe")
            total_duration = self.get_total_duration(self.video_path)
            bitrate = self.get_bitrate(self.video_path)

            command = [
                ffmpeg_path,
                # "ffmpeg", 
                "-i", self.video_path,
                "-g", "1",
                "-c:v", "libx264", "-b:v", f"{bitrate}",
                "-an", output_path, "-y"
            ]

            process = subprocess.Popen(command, stderr=subprocess.PIPE,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)

            for line in iter(process.stderr.readline, ""):
                if not self.running:
                    process.terminate()
                    self.status_signal.emit("Conversion stopped.")
                    return

                if "time=" in line:
                    match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                    if match:
                        elapsed_time = (
                            int(match.group(1)) * 3600 +
                            int(match.group(2)) * 60 +
                            float(match.group(3))
                        )
                        progress = (elapsed_time / total_duration) * 100
                        self.progress_signal.emit(int(progress))

            process.wait()

            if process.returncode == 0:
                self.status_signal.emit("Conversion completed successfully!")
                self.result.emit(output_path)
            else:
                self.status_signal.emit("Error during conversion.")

        except Exception as e:
            self.status_signal.emit(f"Error: {e}")
    
    def stop(self):
        self.running = False


def check_video_format(video_path, time = 1):        
    def extract_first_frames(video_path, time, output_path="video_format_check.mp4"):
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
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return output_path
    
    def analyze_video(video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        cmd = [
            ffprobe_path,
            "-i", video_path,
            "-show_frames",
            "-show_entries", "frame=pict_type",
            "-of", "csv"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        frames = [
            line.split(",")[-1] for line in result.stdout.splitlines()
            if line.startswith("frame") and line.split(",")[-1].strip() != ""
        ]
        return frames

    output_video = extract_first_frames(video_path, time)
    frame_types = analyze_video(output_video)

    os.remove(output_video)
    
    if "P" in frame_types:
        return "P"
    elif "B" in frame_types:
        return "B"
    elif "I" in frame_types:
        return "I"

    