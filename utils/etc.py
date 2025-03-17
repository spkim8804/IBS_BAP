import numpy as np
import sys
import json
import os
import glob

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox
)

def zero_replacing(points_data):
    """
    Replace 0 or None value by nearest value
    """
    
    # Treat 0 as missing (convert 0 to np.nan)
    points_data[points_data == 0] = np.nan
    # print(points_data)
    # Forward fill
    for i in range(1, len(points_data)):
        mask = np.isnan(points_data[i, :])
        points_data[i, mask] = points_data[i - 1, mask]
    
    # Backward fill
    for i in range(len(points_data) - 2, -1, -1):
        mask = np.isnan(points_data[i, :])
        points_data[i, mask] = points_data[i + 1, mask]

    return points_data

def get_AB_AC_angle(xA, yA, xB, yB, xC, yC):
    # Define your points (A, B, C)
    A = np.array([xA, yA])  # Replace xA, yA with the coordinates of point A
    B = np.array([xB, yB])  # Replace xB, yB with the coordinates of point B
    C = np.array([xC, yC])  # Replace xC, yC with the coordinates of point C
    
    # Create vectors AB and AC
    AB = B - A
    AC = C - A
    
    # Calculate the angle using arctan2
    angle = np.arctan2(AC[1], AC[0]) - np.arctan2(AB[1], AB[0])
    
    # Normalize the angle to be between 0 and 2π (0 and 360 degrees)
    angle = np.mod(angle, 2 * np.pi)
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    
    return(angle_degrees)

def calc_min_distance(click_x, click_y, x1, y1, x2, y2):
    # Calculate minimum distance to the edges
    if x1 <= click_x <= x2:  # Within vertical bounds
        dist_top = abs(click_y - y1)  # Top edge
        dist_bottom = abs(click_y - y2)  # Bottom edge
        min_vert_dist = min(dist_top, dist_bottom)
    else:
        min_vert_dist = float("inf")

    if y1 <= click_y <= y2:  # Within horizontal bounds
        dist_left = abs(click_x - x1)  # Left edge
        dist_right = abs(click_x - x2)  # Right edge
        min_horiz_dist = min(dist_left, dist_right)
    else:
        min_horiz_dist = float("inf")

    # Compute the overall minimum distance to the rectangle
    min_distance = min(min_vert_dist, min_horiz_dist)

    # If the click is outside the rectangle, calculate corner distances
    if click_x < x1 or click_x > x2 or click_y < y1 or click_y > y2:
        corner_distances = [
            ((click_x - x1) ** 2 + (click_y - y1) ** 2) ** 0.5,  # Top-left corner
            ((click_x - x2) ** 2 + (click_y - y1) ** 2) ** 0.5,  # Top-right corner
            ((click_x - x1) ** 2 + (click_y - y2) ** 2) ** 0.5,  # Bottom-left corner
            ((click_x - x2) ** 2 + (click_y - y2) ** 2) ** 0.5,  # Bottom-right corner
        ]
        min_distance = min(min_distance, *corner_distances)

    return min_distance

class JsonViewer(QWidget):
    def __init__(self, json_file=None):
        super().__init__()
        self.current_file = json_file  # default JSON file
        self.init_ui()

        if self.current_file:
            self.load_json_file(initial_load=True)

    def init_ui(self):
        self.setWindowTitle(f"{self.current_file}")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.open_button = QPushButton("Open JSON file", self)
        self.open_button.clicked.connect(self.load_json_file)

        self.save_button = QPushButton("Save JSON file", self)
        self.save_button.clicked.connect(self.save_json_file)

        self.text_edit = QTextEdit(self)

        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def load_json_file(self, initial_load=False):
        if not initial_load:  # try to open other json file
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Choose JSON file", "", "JSON Files (*.json);;All Files (*)", options=options
            )
            if not file_path:
                return
            self.current_file = file_path  # 새 파일 경로 저장
            self.setWindowTitle(f"{self.current_file}")

        if self.current_file:  # JSON 파일 경로가 존재할 때
            try:
                with open(self.current_file, "r", encoding="utf-8") as file:
                    json_data = json.load(file)

                pretty_json = json.dumps(json_data, indent=4, ensure_ascii=False)
                self.text_edit.setText(pretty_json)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error occurs during JSON file load:\n{str(e)}")

    def save_json_file(self):
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please open JSON file.")
            return

        try:
            json_text = self.text_edit.toPlainText()
            json_data = json.loads(json_text)  # JSON 유효성 검사

            with open(self.current_file, "w", encoding="utf-8") as file:
                json.dump(json_data, file, indent=4, ensure_ascii=False)

            QMessageBox.information(self, "Save complete", "JSON file is successfully saved.")

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Save failure", "Invalid JSON file. Please check the file.")

def find_images_and_texts_yolo_format(image_root, label_root):
    train_list = []  # (JPG 파일 경로, 대응하는 TXT 파일 경로) 저장 리스트

    # images 폴더 내 train, val 등 하위 폴더 탐색
    for subdir in ["train", "val"]:  # train, val 폴더만 탐색
        image_folder = os.path.join(image_root, subdir)  # 이미지 폴더 경로
        label_folder = os.path.join(label_root, subdir)  # 라벨 폴더 경로

        if not os.path.exists(image_folder) or not os.path.exists(label_folder):
            print(f"폴더 없음: {image_folder} 또는 {label_folder}")
            continue  # 폴더가 없으면 건너뜀

        # 이미지 폴더에서 jpg 파일 찾기
        for file in os.listdir(image_folder):
            if file.lower().endswith(".jpg"):  # 확장자가 jpg인지 확인
                jpg_path = os.path.join(image_folder, file)
                txt_path = os.path.join(label_folder, file.replace(".jpg", ".txt"))  # txt 파일 경로 예상

                # 해당 txt 파일이 존재하는지 확인
                if os.path.exists(txt_path):
                    train_list.append((jpg_path, txt_path))
                else:
                    train_list.append((jpg_path, None))  # txt 파일이 없으면 None 저장

    return train_list

def find_images_and_texts_same_folder(folderPath):
    train_list = []

    for file in os.listdir(folderPath):
        if file.lower().endswith(".jpg"):  # 확장자가 jpg인지 확인
            jpg_path = os.path.join(folderPath, file)
            txt_path = os.path.join(folderPath, file.replace(".jpg", ".txt"))  # txt 파일 경로 예상

            # 해당 txt 파일이 존재하는지 확인
            if os.path.exists(txt_path):
                train_list.append((jpg_path, txt_path))
            else:
                train_list.append((jpg_path, None))  # txt 파일이 없으면 None 저장

    return train_list

def find_json_files(directory):
    json_file_names = []
    for file_path in glob.glob(os.path.join(directory, "*.json")):
        json_file_names.append(file_path)
    return json_file_names