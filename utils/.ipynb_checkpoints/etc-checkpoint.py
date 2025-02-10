import numpy as np
import sys
import json
import os
import glob

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit

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
    def __init__(self, json_file_path):
        super().__init__()

        self.setWindowTitle("JSON viewer")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.setLayout(layout)

        self.load_json(json_file_path)

    def load_json(self, file_path):
        if not os.path.exists(file_path):
            self.text_area.setText(f"No file was found: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

            formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
            self.text_area.setText(formatted_json)

        except Exception as e:
            self.text_area.setText(f"Cannot open the file: {str(e)}")

def find_json_files(directory):
    json_file_names = []
    for file_path in glob.glob(os.path.join(directory, "*.json")):
        json_file_names.append(file_path)
    return json_file_names