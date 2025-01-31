import sys
import numpy as np
import pandas as pd
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QVBoxLayout, QSlider, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *

class Point3DWidget(QOpenGLWidget):
    def __init__(self, parent=None, csv_file="text.csv", config_file="config.json"):
        super().__init__(parent)
        self.time = 0  # 초기 시간
        self.rotation_x = 0  # 마우스 드래그로 X축 회전
        self.rotation_y = 0  # 마우스 드래그로 Y축 회전
        self.zoom = -50.0  # 초기 확대/축소 값
        self.last_mouse_pos = None  # 마우스 위치 저장

        # CSV 파일 로드
        self.data = pd.read_csv(csv_file, header=None).values  # CSV 데이터를 numpy 배열로 변환
        self.num_frames = len(self.data)  # 총 프레임 수
        self.num_points = 9  # 점의 개수
        self.colors = np.random.rand(self.num_points, 3)  # 각 점의 색상 (RGB)

        # 데이터의 x, y, z 범위 계산
        self.data = self.data.reshape(self.num_frames, self.num_points, 3)
        self.x_min, self.x_max = self.data[:, :, 0].min(), self.data[:, :, 0].max()
        self.y_min, self.y_max = self.data[:, :, 1].min(), self.data[:, :, 1].max()
        self.z_min, self.z_max = self.data[:, :, 2].min(), self.data[:, :, 2].max()

        # JSON 파일에서 연결 관계 로드
        with open(config_file, "r") as f:
            config = json.load(f)
        self.lines = config["keypoint"]["line"]

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화
        glClearColor(0.1, 0.1, 0.1, 1.0)  # 배경색 설정
        glEnable(GL_BLEND)  # 투명도 활성화
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # 투명도 설정
        glEnable(GL_POINT_SMOOTH)  # 점 부드럽게
        glPointSize(10)  # 점 크기 설정

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 카메라 위치 조정 (상자 외부에서 바라보기)
        glTranslatef(0.0, 0.0, self.zoom)  # 카메라 위치 조정
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)  # X축 회전
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)  # Y축 회전

        # 투명한 상자 그리기
        self.draw_box()

        # 현재 시간에 해당하는 점 데이터 가져오기
        frame_index = int(self.time) % self.num_frames
        points = self.data[frame_index]

        # 점 그리기
        self.draw_points(points)

        # 점 사이의 선 그리기
        self.draw_lines(points)

    def draw_points(self, points):
        """점 그리기"""
        glBegin(GL_POINTS)
        for i, (x, y, z) in enumerate(points):
            glColor3f(*self.colors[i])  # 색상 설정
            glVertex3f(x, y, z)  # 점 위치
        glEnd()

    def draw_lines(self, points):
        """점 사이의 선 그리기"""
        glColor4f(1.0, 1.0, 1.0, 0.2)  # 흰색, 80% 투명도
        glBegin(GL_LINES)
        for line in self.lines:
            p1, p2 = line
            glVertex3f(*points[p1])  # 첫 번째 점
            glVertex3f(*points[p2])  # 두 번째 점
        glEnd()

    def draw_box(self):
        """데이터 범위에 기반한 투명한 상자 그리기"""
        glColor4f(1.0, 1.0, 1.0, 0.2)  # 흰색, 20% 투명도
        glBegin(GL_LINES)

        # x, y, z 최소/최대값으로 8개의 점 정의
        corners = [
            (self.x_min, self.y_min, self.z_min),
            (self.x_max, self.y_min, self.z_min),
            (self.x_max, self.y_max, self.z_min),
            (self.x_min, self.y_max, self.z_min),
            (self.x_min, self.y_min, self.z_max),
            (self.x_max, self.y_min, self.z_max),
            (self.x_max, self.y_max, self.z_max),
            (self.x_min, self.y_max, self.z_max),
        ]

        # 상자의 12개 모서리 정의
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 아래면
            (4, 5), (5, 6), (6, 7), (7, 4),  # 위면
            (0, 4), (1, 5), (2, 6), (3, 7)   # 옆면 연결
        ]

        for edge in edges:
            for vertex in edge:
                glVertex3f(*corners[vertex])

        glEnd()

    def update_time(self, time):
        """슬라이더 또는 애니메이션으로 시간을 업데이트"""
        self.time = time
        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.rotation_x += dy * 0.5  # Y축 드래그 → X축 회전
            self.rotation_y += dx * 0.5  # X축 드래그 → Y축 회전
            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        """마우스 휠로 확대/축소"""
        delta = event.angleDelta().y() / 120  # 휠 회전값을 얻음
        self.zoom += delta * 0.5  # 확대/축소 크기 조정
        self.zoom = max(-50, min(self.zoom, -5))  # 확대/축소 범위 제한
        self.update()


class PoseVisualizer(QMainWindow):
    def __init__(self, csv_file="text.csv", config_file="./config/config.json"):
        super().__init__()
        self.setWindowTitle("Pose Visualizer")
        self.setGeometry(100, 100, 800, 600)

        # OpenGL 위젯 생성
        self.opengl_widget = Point3DWidget(self, csv_file=csv_file, config_file=config_file)

        # 슬라이더 생성
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(self.opengl_widget.num_frames)
        self.slider.setValue(1)  # 초기 값
        self.slider.valueChanged.connect(self.on_slider_value_changed)

        # 플레이 버튼 생성
        self.play_button = QPushButton("Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)

        # 애니메이션을 위한 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)

        # 레이아웃 설정
        central_widget = QWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(self.opengl_widget)
        layout.addWidget(self.slider)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        layout.addLayout(button_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_slider_value_changed(self, value):
        """슬라이더 값 변경 이벤트"""
        # self.opengl_widget.update_time(value / 1000.0 * self.opengl_widget.num_frames)
        self.opengl_widget.update_time(value)

    def toggle_playback(self, checked):
        """재생/정지 토글"""
        if checked:
            self.play_button.setText("Pause")
            self.timer.start(10)
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def animate(self):
        """애니메이션 프레임 업데이트"""
        current_time = self.slider.value()
        new_time = (current_time + 1) % 1000
        self.slider.setValue(new_time)

# if __name__ == "__main__":
#     # CSV 파일 경로 지정
#     csv_file_path = "adult_6116.csv"

#     app = QApplication.instance()
#     if app is None:
#         app = QApplication(sys.argv)
#     window = MainWindow(csv_file=csv_file_path)
#     window.show()
#     try:
#         app.exec_()
#     except SystemExit:
#         print("[Info] PyQt5 Application exited cleanly.")
#     finally:
#         app.quit()
#         del app
#         print("[Info] QApplication resources have been cleaned up.")
