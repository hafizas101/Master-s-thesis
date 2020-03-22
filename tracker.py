
# coding: utf-8

# # Import libraries and define functions

# In[15]:

from tracker_help import *
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import time, glob, os, natsort, sys, cv2, os, json, math, argparse
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.utils.linear_assignment_ import linear_assignment
from numpy import dot
from scipy.linalg import inv, block_diag


labelsPath = os.path.join(os.getcwd(), "yolo_files/classes.names")
weightsPath = os.path.join(os.getcwd(), "yolo_files/yolov3.weights")
configPath = os.path.join(os.getcwd(), "yolo_files/yolov3.cfg")
confidence_t = 0.5
threshold_t = 0.3
img_size = 416

annot_path = os.path.join(os.getcwd(), "MVI_40852.xml")
img_folder = os.path.join(os.getcwd(), "MVI_40852/*")
file_paths = glob.glob(img_folder)
sorted_file_paths = natsort.natsorted(file_paths, reverse=False)


maxDisappeared = 10  # no.of consecutive unmatched detection before a track is deleted
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.8
font_color = (255, 0, 0)
output_FPS = 10  # Frames per second of output video
max_objects = 50 # Maximum number of vehicles that can be in a frame at some time. Should not exceed this value.


parser = argparse.ArgumentParser(description='Tracking using YOLOv3 or Mask RCNN Benchmark')
parser.add_argument('--method', type=str,    help='Write name of method (yolo or mask)')
args = parser.parse_args()
method = args.method


# # Detection of vehicles

# In[17]:


total_pred_car = 0
pred_car = []
total_pred_bus = 0
pred_bus = []

img_id = []
tree = ET.parse(annot_path)
root = tree.getroot()
ignored_boxes = tree.findall('ignored_region')
images = []

for i, f in enumerate(sorted_file_paths):
    img_id.append(i+1)
    img = cv2.imread(f, 1)
    
    for child in ignored_boxes[0]:
        aa = child.attrib
        x1 = int(float(aa.get('left')))
        y1 = int(float(aa.get('top')))
        w = int(float(aa.get('width')))
        h = int(float(aa.get('height')))
        x2 = x1+w
        y2 = y1+h

        for j in range (h):
            for k in range (w):
                img[y1+j, x1+k] = [255, 255, 255]

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if method=="yolo":   
        list_car, list_bus, image = yolov3_process(img)
    elif method=="mask":
        list_car, list_bus, image = maskrcnn_benchmark_process(img)
    images.append(image)
#         dict_car, dict_bus = maskrcnn_process(img)
    total_pred_car = total_pred_car + len(list_car)
    pred_car.append(list_car)
    
    total_pred_bus = total_pred_bus + len(list_bus)
    pred_bus.append(list_bus)


print("Number of cars detected: "+str(total_pred_car))
print("Number of buses detected: "+str(total_pred_bus))


# dict_pred_car = {'image_id': img_id, 'frame_predictions': pred_car, 'frames': images}
# dict_pred_bus = {'image_id': img_id, 'frame_predictions': pred_bus, 'frames': images}



# # Centroid Tracker ( Gives noisy trajectory )

ct = CentroidTracker()

labelled_frames = []
num = len(pred_car)
total_ids = np.linspace(0,max_objects-1, max_objects)
ids = []
XX = np.zeros(shape = (len(images), max_objects), dtype=int)
YY = np.zeros(shape = (len(images), max_objects), dtype = int)

if len(pred_car) == len(pred_bus):
    counts = []

    for i in range(len(pred_car)):
        comp_list = pred_car[i] + pred_bus[i]
        frame = images[i].copy()
        objects = ct.update(comp_list)
        c = 0
        for (objectID, centroid) in objects.items():
            ids.append(objectID)
            c = c+1
            XX[i,objectID] = centroid[0]
            YY[i,objectID] = centroid[1]
            

length_path = len(images)
labelled = []
frames = images.copy()
pts_list = []
colors = []
for i in range(XX.shape[1]):
    pts_list.append(deque(maxlen=length_path))
    colors.append((np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)))
for i in range (len(images)):
    img = frames[i].copy()
    for j in range(XX.shape[1]):
        pts_list[j].appendleft((XX[i,j], YY[i,j]))       
        for i in range(1, len(pts_list[j])):
            if pts_list[j][i - 1] is None or pts_list[j][i] is None:
                continue
            thickness = 4
#             thickness = int(np.sqrt(length_path / float(i + 1)) * 2)
            d = calculateDistance(pts_list[j][i - 1][0], pts_list[j][i-1][1], pts_list[j][i][0], pts_list[j][i][1], )
            if d<100:
                cv2.line(img, pts_list[j][i - 1], pts_list[j][i], colors[j], thickness)

#     cv2.imshow("img", img)
#     cv2.waitKey(0)
    labelled.append(img)
# cv2.destroyAllWindows()


output_video_path = os.path.join(os.getcwd(), "centroid_tracker.mp4")
convert_frames_to_video(labelled, output_video_path)


global tracker_list
global max_age
global min_hits
global track_id_list
global debug

# Global variables to be used by funcitons of VideoFileClop
max_age = maxDisappeared

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID (0-49)
track_id_list= deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])

debug = True

xx_kalman = np.zeros(shape=(len(images), max_objects))
yy_kalman = np.zeros(shape=(len(images), max_objects))
length_path = len(images)
labelled = []

if len(pred_car) == len(pred_bus):
    print("Initializing Kalman filter based tracker")

    for i, car in enumerate(pred_car):
        z_box = car + pred_bus[i]
        img = images[i].copy()
        x_box =[]
        

        if len(tracker_list) > 0:
            for trk in tracker_list:
                x_box.append(trk.box)

        matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  

        # Deal with matched detections     
        if matched.size >0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0

        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
                tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx


        # The list of tracks to be annotated  
        good_tracker_list =[]
        for trk in tracker_list:
            if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
                good_tracker_list.append(trk)
                x_cv2 = trk.box
                    
                center_x = int((x_cv2[0] + x_cv2[2]) / 2.0)
                center_y = int((x_cv2[1] + x_cv2[3]) / 2.0)
                xx_kalman[i,trk.id] = center_x
                yy_kalman[i,trk.id] = center_y
                
        deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  

        for trk in deleted_tracks:
            track_id_list.append(trk.id)

        tracker_list = [x for x in tracker_list if x.no_losses<=max_age]



# In[23]:


labelled = []
pts_list = []
colors = []
for i in range(max_objects):
    pts_list.append(deque(maxlen=length_path))
    colors.append((np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)))

for k in range(xx_kalman.shape[0]):
    img = images[k].copy()
    for j in range(xx_kalman.shape[1]):
        pts_list[j].appendleft((int(xx_kalman[k,j]), int(yy_kalman[k,j])))       
        for i in range(1, len(pts_list[j])):
            if pts_list[j][i - 1] is None or pts_list[j][i] is None:
                continue
            thickness = 4
            d = int(calculateDistance(pts_list[j][i - 1][0], pts_list[j][i-1][1], pts_list[j][i][0], pts_list[j][i][1]))
            if d<100:
                cv2.line(img, pts_list[j][i - 1], pts_list[j][i], colors[j], thickness)
                
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
    labelled.append(img)
    
# cv2.destroyAllWindows()

# In[24]:


output_video_path = os.path.join(os.getcwd(), "kalmanFilter_tracker.mp4")
convert_frames_to_video(labelled, output_video_path)

