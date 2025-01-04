import cv2

def export_frame_image(frame_data):
    frame, frame_id, image_path = frame_data
    cv2.imwrite(image_path, frame)