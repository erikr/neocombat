import os
import sys
import cv2
import math
from ultralytics import YOLO


def box_center(coords):
    [left, top, right, bottom] = coords
    return [(left + right) // 2, (top + bottom) // 2]


def closest_box(boxes, coords):
    distance = []
    center = box_center(coords)
    for box in boxes:
        box_center_coord = box_center(box.xyxy[0].numpy().astype(int))
        distance.append(math.dist(box_center_coord, center))
    return boxes[distance.index(min(distance))]


def adjust_box_size(coords, box_width, box_height):
    [center_x, center_y] = box_center(coords)
    return [
        center_x - box_width // 2,
        center_y - box_height // 2,
        center_x + box_width // 2,
        center_y + box_height // 2,
    ]


def adjust_boundaries(coords, screen):
    [left, top, right, bottom] = coords
    [width, height] = screen
    left, top, right, bottom = (
        max(left, 0),
        max(top, 0),
        min(right, width),
        min(bottom, height),
    )
    return [int(coord) for coord in [left, top, right, bottom]]


def ensure_fixed_crop_dimensions(coords, screen, crop_width, crop_height):
    [center_x, center_y] = box_center(coords)
    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(screen[0], center_x + crop_width // 2)
    bottom = min(screen[1], center_y + crop_height // 2)
    return [int(coord) for coord in [left, top, right, bottom]]


def main():
    infer = True
    frame_start = 1800
    frame_end = 2100
    root_dir = os.path.expanduser("~/data")
    fname = "C0108x"
    w_i = 0
    h_i = 0
    crop_width = 400
    crop_height = 400
    crop_coords = [w_i, h_i, w_i + crop_width, h_i + crop_height]

    fpath_video_input = os.path.join(root_dir, f"{fname}.mp4")
    if not os.path.exists(fpath_video_input):
        print("Input video file not found.")
        sys.exit(1)

    ext = "mp4"
    fpath_video_output = os.path.join(root_dir, f"{fname}-output2.{ext}")

    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(fpath_video_input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video metadata: {fps} fps, {width} width x {height} height")

    [left, top, right, bottom] = adjust_boundaries(crop_coords, [width, height])
    last_coords = [left, top, right, bottom]
    last_box_coords = last_coords
    box_width = right - left
    box_height = bottom - top

    if (crop_width > width) or (crop_height > height):
        raise ValueError("Crop dimensions are greater than the input video dimensions.")

    out = cv2.VideoWriter(
        fpath_video_output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_width, crop_height),
    )
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > frame_end:
            break
        else:
            if frame_count < frame_start:
                print(f"Skipping frame {frame_count}")
            else:
                print(f"Processing frame {frame_count}")
                results = model.predict(source=frame, conf=0.5, iou=0.1)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    closest = closest_box(boxes, last_box_coords)
                    last_box_coords = closest.xyxy[0].numpy().astype(int)

                    # Ensure fixed crop dimensions
                    new_coords = ensure_fixed_crop_dimensions(
                        last_box_coords, [width, height], crop_width, crop_height
                    )

                    [left, top, right, bottom] = new_coords
                    frame = frame[top:bottom, left:right]

                out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
    print("Code successful. Exiting.")
