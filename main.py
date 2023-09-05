import os
import cv2 
from ultralytics import YOLO 


def boxCenter(coords):
    [left, top, right, bottom] = coords
    return [(left + right) / 2, (top + bottom) / 2]


def closestBox(boxes, coords):
    distance = []
    center = boxCenter(coords)
    for box in boxes:
        boxCent = boxCenter(box.xyxy[0].numpy().astype(int))
        distance.append(math.dist(boxCent, center))
    return boxes[distance.index(min(distance))]


def adjustBoxSize(coords, box_width, box_height):
    [centerX, centerY] = boxCenter(coords)
    return [
        centerX - box_width / 2,
        centerY - box_height / 2,
        centerX + box_width / 2,
        centerY + box_height / 2,
    ]


def adjustBoundaries(coords, screen):
    [left, top, right, bottom] = coords
    [width, height] = screen
    if left < 0:
        right = right - left
        left = 0
    if top < 0:
        bottom = bottom - top
        top = 0
    if right > width:
        left = left - (right - width)
        right = width
    if bottom > height:
        top = top - (bottom - height)
        bottom = height
    return [round(left), round(top), round(right), round(bottom)]


def main():
    # Set path to source file
    fname = "C0108x.mp4"
    fileSource = os.path.expanduser(f"~/{fname}")
    
    # Set path to video file in which processed video will be saved
    fileTarget = os.path.expanduser("~/{fname}-output.mp4")

    # Load pre-trained model
    model = YOLO("yolov8s.pt")

    # coordinates of the cropping box we will start with, this cropping box will follow our object
    cropCoords = [
        100,
        100,
        500,
        500,
    ]

    vidCapture = cv2.VideoCapture(fileSource)
    fps = vidCapture.get(cv2.CAP_PROP_FPS)
    totalFrames = vidCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cropCoords:
        [box_left, box_top, box_right, box_bottom] = [0, 0, width, height]
    else:
        [box_left, box_top, box_right, box_bottom] = cropCoords
        if box_left < 0:
            box_left = 0
        if box_top < 0:
            box_top = 0
        if (box_right) > width:
            box_right = width
        if box_bottom > height:
            box_bottom = height
    lastCoords = [box_left, box_top, box_right, box_bottom]
    lastBoxCoords = lastCoords
    box_width = box_right - box_left
    box_height = box_bottom - box_top

    # Open an output video stream
    outputWriter = cv2.VideoWriter(
        fileTarget, cv2.VideoWriter_fourcc(*"MPEG"), fps, (box_width, box_height)
    )

    # Read a frame from the input video stream as an image. Read the next frame from the input video stream until the end of the video.
    frameCounter = 1
    while True:
        r, im = vidCapture.read()

        if not r:
            print("Video Finished!")
            break

        print("Frame: " + str(frameCounter))
        frameCounter = frameCounter + 1

    results = model.predict(
        source=im, conf=0.5, iou=0.1
    )  # request for the YOLO model to find objects, you can see the documentation on the YOLO model for params
    boxes = results[0].boxes  # boxes are coordinates of objects YOLO has found
    box = closestBox(
        boxes, lastBoxCoords
    )  # returns the best box - closest to the last one
    lastBoxCoords = (
        box.xyxy[0].numpy().astype(int)
    )  # converts the PyTorch Tensor into box coordinates and saves for the next iteration

    # Crop the image around the object
    newCoords = adjustBoxSize(
        box.xyxy[0].numpy().astype(int), box_width, box_height
    )  # since the area YOLO has found for the object depends on the object but not on the cropping area we need to convert the area of the object to the cropping area
    newCoords = adjustBoundaries(
        newCoords, [width, height]
    )  # don't allow to get the cropping area go out of video screen edges
    [box_left, box_top, box_right, box_bottom] = newCoords
    imCropped = im[box_top:box_bottom, box_left:box_right]  # cropping the image

    # Add the cropped image to the output video stream
    outputWriter.write(
        imCropped
    )  # writing the cropped image as the new frame into the output video stream

    # Close input and output video streams
    vidCapture.release()
    outputWriter.release()


if __name__ == "__main__":
    # main()

    print("Successfully ran main.py! Exiting.")
