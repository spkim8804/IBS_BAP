import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Video Player with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # UI 구성
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)

        self.open_button = QPushButton("🎥 Open Video")
        self.play_button = QPushButton("▶ Play")
        self.pause_button = QPushButton("⏸ Pause")
        self.stop_button = QPushButton("⏹ Stop")

        self.open_button.clicked.connect(self.open_video)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 비디오 관련 변수
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

    def open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print("[Error] Cannot open video file.")
            else:
                print("[Info] Video loaded successfully.")

    def play_video(self):
        if self.cap and not self.is_playing:
            self.is_playing = True
            self.timer.start(30)  # 약 30ms마다 프레임 업데이트 (FPS 30 가정)

    def pause_video(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()

    def stop_video(self):
        if self.cap:
            self.is_playing = False
            self.timer.stop()
            self.cap.release()
            self.video_label.clear()

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                # OpenCV는 BGR 포맷, PyQt는 RGB 포맷 필요
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
            else:
                self.stop_video()
                print("[Info] Video playback finished.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
