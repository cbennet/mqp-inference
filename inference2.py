import cv2
import numpy as np
import time
import math
from utils import *
from darknet import Darknet

CONFIDENCE = 0.8
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "model.cfg"
weights_path = "model.weights"
name_path = "model.names"
font_scale = 1
thickness = 1
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

#net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
m = Darknet(config_path)
m.load_weights(weights_path)
class_names = load_class_names(name_path)
# wired webcam
pic = cv2.VideoCapture(0)

# ipcam
# pic = cv2.VideoCapture(<stream url>)

# default image
img = pic

# change resolution
# img = pic.set(3,680)
# img = pic.set(4,480)

while True:
    _, image = img.read()

    # detect the objects
    boxes = detect_objects(m, img, iou_threshold, nms_threshold)

    # plot the image with the bounding boxes and corresponding object class labels
    plot_boxes(original_image, boxes, class_names, plot_labels=True)

    if ord("q") == cv2.waitKey(1):
        break

img.release()
cv2.destroyAllWindows()