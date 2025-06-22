import cv2
import numpy as np
import os
import uuid

# COCO 關鍵點定義 (17 個點的順序)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle"
]
# COCO 骨架連接關係 (基於 COCO_KEYPOINTS 的索引)
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_pose_on_frame(frame, predictions, min_score_thresh=0.5, point_radius=5, line_thickness=2):
    """
    在單個影片幀上繪製姿態關鍵點和骨架。
    """
    if not predictions:
        return frame

    for person_prediction in predictions:
        keypoints_coords = person_prediction.get('keypoints', [])
        keypoint_scores = person_prediction.get('keypoint_scores', [])

        if not keypoints_coords or not keypoint_scores:
            continue

        keypoints_data = []
        for i, (x, y) in enumerate(keypoints_coords):
            score = keypoint_scores[i] if i < len(keypoint_scores) else 0.0
            keypoints_data.append((int(x), int(y), score))

        for kp_idx, (x, y, score) in enumerate(keypoints_data):
            if score > min_score_thresh:
                cv2.circle(frame, (x, y), point_radius, (0, 255, 0), -1)

        for connection in COCO_CONNECTIONS:
            start_kp_idx = connection[0]
            end_kp_idx = connection[1]

            if start_kp_idx < len(keypoints_data) and end_kp_idx < len(keypoints_data):
                x1, y1, s1 = keypoints_data[start_kp_idx]
                x2, y2, s2 = keypoints_data[end_kp_idx]

                if s1 > min_score_thresh and s2 > min_score_thresh:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)
    return frame

def render_video_with_pose(video_path: str, api_response_json: dict, output_dir: str) -> str:
    """
    將姿態偵測結果渲染到原始影片的每一幀上，並將其保存為新的影片檔案。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_video_id = str(uuid.uuid4())[:8]
    # 確保輸出檔案擴展名是 .mp4
    processed_video_filename = f"{processed_video_id}_pose_rendered.mp4"
    processed_video_path = os.path.join(output_dir, processed_video_filename)

    # 嘗試使用 'avc1' 編解碼器以提高兼容性
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Changed from 'mp4v' to 'avc1'
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {processed_video_path}. Check codec or path permissions.")
        print("Common reasons: Missing FFmpeg or incompatible codec. Try 'mp4v' or ensure FFmpeg is installed.")
        cap.release()
        return ""

    api_frames_data = {frame_data['frame_idx']: frame_data for frame_data in api_response_json.get('frames', [])}

    current_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions_raw = api_frames_data.get(current_frame_idx, {}).get('predictions', [])

        predictions_for_current_frame = []
        if predictions_raw and isinstance(predictions_raw, list) and predictions_raw[0] and isinstance(predictions_raw[0], list):
            predictions_for_current_frame = predictions_raw[0]
        elif predictions_raw and isinstance(predictions_raw, list) and predictions_raw[0] and isinstance(predictions_raw[0], dict):
            predictions_for_current_frame = predictions_raw

        rendered_frame = draw_pose_on_frame(
            frame.copy(),
            predictions_for_current_frame
        )

        out.write(rendered_frame)
        current_frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished rendering {current_frame_idx} frames and saved to {processed_video_path}.")
    return processed_video_path
