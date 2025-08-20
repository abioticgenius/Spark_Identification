import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Paths
model_path = '/content/drive/MyDrive/Panto_2.v1i.yolov8-obb/runs/detect/train2/weights/best.pt'
input_video_path = '/content/drive/MyDrive/Contact_Point_Detection_2.v2i.yolov8/spark_4.mp4'
output_video_path = '/content/drive/MyDrive/Contact_Point_Detection_2.v2i.yolov8/spark_4_output.mp4'
csv_output_path = '/content/drive/MyDrive/Contact_Point_Detection_2.v2i.yolov8/spark_4_data.csv'

# Initialize YOLO model
model = YOLO(model_path)

# Initialize video capture and writer
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Variables for tracking movements and previous positions
frame_count = 0
previous_pantograph_center = None
horizontal_movement = 0
vertical_movement = 0

# CSV file setup
with open(csv_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp (s)', 'Horizontal Movement (px)', 'Vertical Movement (px)', 'Spark Detected'])

# Utility functions
def find_contact_points(line, box):
    x1, y1, x2, y2 = box
    box_lines = [
        [(x1, y1), (x2, y1)],
        [(x2, y1), (x2, y2)],
        [(x2, y2), (x1, y2)],
        [(x1, y2), (x1, y1)]
    ]
    contact_points = []
    for box_line in box_lines:
        intersection = line_intersection(line, box_line)
        if intersection and x1 <= intersection[0] <= x2 and y1 <= intersection[1] <= y2:
            contact_points.append(intersection)
    return contact_points

def line_intersection(line1, line2):
    def line_eq(p1, p2):
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0] * p2[1] - p2[0] * p1[1]
        return A, B, -C

    p1, p2 = line1
    q1, q2 = line2
    L1 = line_eq(p1, p2)
    L2 = line_eq(q1, q2)
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    return None

def get_color(index):
    np.random.seed(index)
    return tuple(np.random.randint(0, 256, 3).tolist())

def detect_sparks(frame, bbox, frame_brightness):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if frame_brightness > 150:
        _, thresh = cv2.threshold(gray_roi, 240, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(gray_roi, 220, 255, cv2.THRESH_BINARY)

    mask = cv2.inRange(roi, (200, 200, 200), (255, 255, 255))
    filtered_thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(filtered_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spark_detected = False

    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 50:
            cv2.rectangle(frame, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 255, 0), 2)
            spark_detected = True

    return spark_detected

# Main loop for processing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    results = model(frame)

    pantograph_box = None
    pantobar_box = None
    cable_boxes = []
    contact_points_distances = []
    spark_detected = False

    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id == 2:  # Pantograph class ID
                pantograph_box = (x1, y1, x2, y2)
            elif class_id == 1:  # Pantobar class ID
                pantobar_box = (x1, y1, x2, y2)
            elif class_id == 0:  # Cable class ID
                cable_boxes.append((x1, y1, x2, y2))

    if pantograph_box is not None:
        px1, py1, px2, py2 = pantograph_box
        pantograph_height = py2 - py1
        pantograph_height_text = f'Pantograph Height: {pantograph_height}px'
        cv2.putText(frame, pantograph_height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        pantograph_center = ((px1 + px2) // 2, (py1 + py2) // 2)

        if previous_pantograph_center is not None:
            dx = pantograph_center[0] - previous_pantograph_center[0]
            dy = pantograph_center[1] - previous_pantograph_center[1]
            horizontal_movement += dx
            vertical_movement += dy

        previous_pantograph_center = pantograph_center

        # Detect sparks within the pantograph box
        spark_detected = detect_sparks(frame, pantograph_box, frame_brightness)

    if pantobar_box is not None:
        bx1, by1, bx2, by2 = pantobar_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        pantobar_top_line = [(bx1, by1), (bx2, by1)]
        cv2.line(frame, pantobar_top_line[0], pantobar_top_line[1], (255, 0, 0), 2)
        center_point = ((bx1 + bx2) // 2, by1)

        for i, cable_box in enumerate(cable_boxes):
            cx1, cy1, cx2, cy2 = cable_box
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
            contact_points = find_contact_points(pantobar_top_line, cable_box)

            if contact_points:
                mid_x = sum(p[0] for p in contact_points) // len(contact_points)
                mid_y = sum(p[1] for p in contact_points) // len(contact_points)
                midpoint = (mid_x, mid_y)
                color = get_color(i)
                cv2.circle(frame, midpoint, 6, color, -1)

                line_length = bx2 - bx1
                distance_meters = (np.linalg.norm(np.array(midpoint) - np.array(center_point)) / line_length) * 3
                contact_points_distances.append((distance_meters, color))

        for i, (distance, color) in enumerate(contact_points_distances):
            text = f'{distance:.2f} m'
            cv2.putText(frame, text, (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Record data to CSV every second
    if spark_detected or frame_count % fps == 0:
        timestamp = frame_count / fps
        with open(csv_output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, horizontal_movement, vertical_movement, spark_detected])
        horizontal_movement = 0
        vertical_movement = 0

    # Write frame to output video
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
