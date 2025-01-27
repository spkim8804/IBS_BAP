import sys
import os
import subprocess
import re
import json
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QSlider, QHBoxLayout, QListWidget, QListWidgetItem,
    QComboBox, QScrollArea, QMenuBar, QAction, QMessageBox, QDialog, QProgressBar
)

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
    progress_signal = pyqtSignal(str)
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

    def get_avg_frame_rate(self, video_path):
        ffprobe_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffprobe.exe")
        cmd = [
            ffprobe_path,
            "-i", video_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "json"
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            if "streams" in data and len(data["streams"]) > 0:
                avg_frame_rate = data["streams"][0].get("avg_frame_rate", "N/A")
                num, den = map(int, avg_frame_rate.split('/'))
                return num / den if den != 0 else 0
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def run(self):
        directory, filename = os.path.split(self.video_path)
        filename, ext = os.path.splitext(filename)
        output_path = f"{directory}/{filename}_iframe{ext}"
        
        bitrate = self.get_bitrate(self.video_path)
        framerate = str(self.get_avg_frame_rate(self.video_path))
        
        try:
            ffmpeg_path = os.path.join(os.getcwd(), "utils\\ffmpeg\\ffmpeg.exe")
            total_duration = self.get_total_duration(self.video_path)
            bitrate = self.get_bitrate(self.video_path)

            command = [
                ffmpeg_path,
                # "ffmpeg", 
                "-i", self.video_path,
                "-r", framerate,
                "-g", "1",
                "-c:v", "libx264",
                "-b:v", f"{bitrate}",
                "-an",
                output_path, "-y"
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
                        progress = f"{((elapsed_time / total_duration) * 100):.2f}"
                        self.progress_signal.emit(progress)

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

class VideoConverterWindow(QWidget):
    conversion_complete_signal = pyqtSignal(str)
    
    def __init__(self, input_file):
        super().__init__()
        self.setWindowTitle("Video Converter")
        self.setGeometry(100, 100, 200, 100)

        self.input_file = input_file
        
        self.init_gui()

    def init_gui(self):
        layout = QVBoxLayout()

        # Progress Bar
        self.progress_label = QLabel("Progress: 0.00%")
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        # Start and Stop Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Conversion")
        self.start_button.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Conversion")
        self.stop_button.clicked.connect(self.stop_conversion)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_conversion(self):
        self.worker = ConvertVideoToIframe(self.input_file)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.status_signal.connect(self.update_status)
        self.worker.result.connect(self.on_conversion_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker.start()

    def stop_conversion(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.close()

    def update_progress(self, progress):
        self.progress_bar.setValue(int(float(progress)))
        self.progress_label.setText(f"Progress: {progress}%")

    def update_status(self, status):
        self.progress_label.setText(status)

    def on_conversion_finished(self, video_path):
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.conversion_complete_signal.emit(video_path)
        self.close()
    