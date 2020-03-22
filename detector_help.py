import numpy as np
import time, glob, os, sys, cv2, os, torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


labelsPath = os.path.join(os.getcwd(), "yolo_files/classes.names")
weightsPath = os.path.join(os.getcwd(), "yolo_files/yolov3.weights")
configPath = os.path.join(os.getcwd(), "yolo_files/yolov3.cfg")
# Recommended parameters by authors. You can play with it.
confidence_t = 0.5
threshold_t = 0.3
img_size = 416


cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml")
cfg.merge_from_list([])
cfg.freeze()
# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(cfg,confidence_threshold=0.7,show_mask_heatmaps=False,masks_per_dim=2,min_image_size=800)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(1)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap



def maskrcnn_benchmark_process(image):
    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)
    labels = top_predictions.get_field("labels")
    scores = top_predictions.get_field("scores")
    boxes = top_predictions.bbox

    locations_car = []
    confi_car = []

    locations_bus = []
    confi_bus = []

    for i, box in enumerate (boxes):
        if coco_demo.CATEGORIES[labels[i]] == "car":
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            location = top_left + bottom_right
            s = scores[i].tolist()

            locations_car.append(location)
            confi_car.append(s)
        elif coco_demo.CATEGORIES[labels[i]] == "bus":
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            location = top_left + bottom_right
            s = scores[i].tolist()

            locations_bus.append(location)
            confi_bus.append(s)

    dictionary_car = {'confidences': confi_car, 'locations': locations_car}
    dictionary_bus = {'confidences': confi_bus, 'locations': locations_bus}

    return dictionary_car, dictionary_bus




def yolov3_process(image):
    LABELS = open(labelsPath).read().strip().split('\n')
    names = []
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    # image should be a 3d numpy array
    (H, W) = image.shape[:2]

    # Let's apply Yolo dectector using pretrained weights
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (img_size, img_size),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
#     print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Let us assign class labels to the objects from layerOutputs
    boxes = []
    confidences = []
    classIDs = []

    classname = []
    list_of_vehicles = ["car", "bus"]
    locations_car = []
    confi_car = []

    locations_bus = []
    confi_bus = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_t:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_t,threshold_t)

    # Let's make output image and store class labels in a list
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            if (LABELS[classIDs[i]] == "car"):
                confi_car.append(confidences[i])
                locations_car.append([x, y, x+w, y+h])
            elif (LABELS[classIDs[i]] == "bus"):
                confi_bus.append(confidences[i])
                locations_bus.append([x, y, x+w, y+h])
                
    dictionary_car = {'confidences': confi_car, 'locations': locations_car}
    dictionary_bus = {'confidences': confi_bus, 'locations': locations_bus}

    return dictionary_car, dictionary_bus
