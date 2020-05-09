#!/usr/bin/env python
# coding: utf-8

# # Import libraries and define functions

# In[1]:

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import time, glob, os, natsort, sys, cv2, os, json, math, torch
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.utils.linear_assignment_ import linear_assignment
from numpy import dot
from scipy.linalg import inv, block_diag
from statsmodels.tsa.arima.model import ARIMA
import copy

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml")
cfg.merge_from_list([])
cfg.freeze()
# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(cfg,confidence_threshold=0.7,show_mask_heatmaps=False,masks_per_dim=2,min_image_size=800)



confidence_t = 0.5
threshold_t = 0.3
img_size = 416

annot_path = os.path.join(os.getcwd(), "MVI_40852.xml")
img_folder = os.path.join(os.getcwd(), "MVI_40852/*")
file_paths = glob.glob(img_folder)
sorted_file_paths = natsort.natsorted(file_paths, reverse=False)


maxDisappeared = 5  # no.of consecutive unmatched detection before a track is deleted
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = (255, 0, 0)
output_FPS = 10  # Frames per second of output video
max_objects = 50 # Maximum number of vehicles that can be in a frame at some time. Should not exceed this value.


# In[2]:


def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist



def convert_frames_to_video(frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    height, width, channels = frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, output_FPS, (width, height))
    for i, ff in enumerate (frames):
        out.write(ff)
    out.release
    cv2.destroyAllWindows()
    return


def maskrcnn_benchmark_process(image):
    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)
    labels = top_predictions.get_field("labels")
    scores = top_predictions.get_field("scores")
    boxes = top_predictions.bbox

    locations_car = []

    locations_bus = []


    for i, box in enumerate (boxes):
        if coco_demo.CATEGORIES[labels[i]] == "car":
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            location = top_left + bottom_right
            locations_car.append(location)
            x = top_left[0]
            y = top_left[1]
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
            cv2.rectangle(image, (x-1, y-15), (x+w+1, y), (0,255,0), -1, 1)
            # Output the labels that show the x and y coordinates of the bounding box center.
            text = "cars"
            cv2.putText(image,text,(x,y-3), font, font_size, font_color, 1, cv2.LINE_AA)

        elif coco_demo.CATEGORIES[labels[i]] == "bus":
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            location = top_left + bottom_right
            s = scores[i].tolist()
            locations_bus.append(location)
            x = top_left[0]
            y = top_left[1]
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
            cv2.rectangle(image, (x-1, y-15), (x+w+1, y), (0,255,0), -1, 1)
            # Output the labels that show the x and y coordinates of the bounding box center.
            text = "cars"
            cv2.putText(image,text,(x,y-3), font, font_size, font_color, 1, cv2.LINE_AA)


    return locations_car, locations_bus, image




# In[3]:


class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id 
        self.box = [] # list to store the coordinates for a bounding box 
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)
        
        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state=[] 
        self.dt = 0.2   # time interval
        
        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])
        
        # Measurement matrix, assuming we can only measure the coordinates
        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        
        
        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L*np.ones(8))
        
        
        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
        # Initialize the measurement covariance
        self.R_scaler = 1
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)
        
        
    def update_R(self):   
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
        
        
    def kalman_filter(self, z): 
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates 
                                     #(pixel values)
        
    def predict_only(self):  
        '''
        Implment only the predict stage. This is used for unmatched detections and unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)


def box_iou2(boxA, boxB):
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





def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
            IOU_mat[t,d] = box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

    def overlap(x1,w1,x2,w2):
        l1 = x1 - w1 / 2.;
        l2 = x2 - w2 / 2.;
        left = max(l1, l2)
        r1 = x1 + w1 / 2.;
        r2 = x2 + w2 / 2.;
        right = min(r1, r2)
        return right - left;

    def box_intersection(a, b):
        w = overlap(a.x, a.w, b.x, b.w);
        h = overlap(a.y, a.h, b.y, b.h);
        if w < 0 or h < 0: return 0;
        area = w * h;
        return area;

    def box_union(a, b):
        i = box_intersection(a, b);
        u = a.w * a.h + b.w * b.h - i;
        return u;

    def box_iou(a, b):
        return box_intersection(a, b) / box_union(a, b);
    
     


# In[4]:


class CentroidTracker():
	def __init__(self, maxDisappeared=maxDisappeared):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# for (i, (startX, startY, endX, endY)) in enumerate(rects):
		# 	# use the bounding box coordinates to derive the centroid
		# 	cX = int((startX + endX) / 2.0)
		# 	cY = endY
		# 	inputCentroids[i] = (cX, cY)


		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects


# # Detection of vehicles

# In[13]:


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

#     img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    list_car, list_bus, image = maskrcnn_benchmark_process(img)
    images.append(img)
#         dict_car, dict_bus = maskrcnn_process(img)
    total_pred_car = total_pred_car + len(list_car)
    pred_car.append(list_car)
    
    total_pred_bus = total_pred_bus + len(list_bus)
    pred_bus.append(list_bus)


print("Number of cars detected: "+str(total_pred_car))
print("Number of buses detected: "+str(total_pred_bus))

cv2.destroyAllWindows()

dict_pred_car = {'image_id': img_id, 'frame_predictions': pred_car, 'frames': images}
dict_pred_bus = {'image_id': img_id, 'frame_predictions': pred_bus, 'frames': images}


# # Ground truth tracker

# In[14]:


xx_ground = np.zeros(shape=(len(images), max_objects))
yy_ground = np.zeros(shape=(len(images), max_objects))

tree = ET.parse(annot_path)
data = []
frames = tree.findall('frame')
for i, f in enumerate (frames):
    ground_boxes = f.findall('target_list/target')
    idd = []
    x = []
    y = []
    for j, child in enumerate(ground_boxes):
        idd.append(int(child.attrib['id']))
        box = child.findall('box')
        attribute = child.findall('attribute')
        aa = box[0].attrib
        x1 = int(float(aa.get('left')))
        y1 = int(float(aa.get('top')))
        w = int(float(aa.get('width')))
        h = int(float(aa.get('height')))
        ground_x = int(x1 + w/2)
        ground_y = int(y1 + h/2)
        x.append(ground_x)
        y.append(ground_y)
        xx_ground[i,j] = ground_x
        yy_ground[i,j] = ground_y        

    di = {'id': idd, 'x': x, 'y': y}
    data.append(di)
        

        


# # Centroid Tracker and Moving Average ( Gives noisy trajectory )

# In[15]:


beta = 0.1
v = 0

ct = CentroidTracker()

labelled_frames = []
num = len(pred_car)
total_ids = np.linspace(0,max_objects-1, max_objects)
XX = np.zeros(shape = (len(images), max_objects), dtype=int)
YY = np.zeros(shape = (len(images), max_objects), dtype = int)
yy_moving_average = np.zeros(shape = (len(images), max_objects), dtype = int)
dd = []
dd_2 = []

if len(pred_car) == len(pred_bus):
    counts = []

    for i in range(len(images)):
        comp_list = pred_car[i] + pred_bus[i]
        frame = images[i].copy()
        objects = ct.update(comp_list)
        db = data[i]
        xs = db['x']
        ys = db['y']
        idss = db['id']
        for (objectID, centroid) in objects.items(): 
            XX[i,objectID] = centroid[0]
            YY[i,objectID] = centroid[1]
            v = beta*v + (1-beta)*centroid[1]
            yy_moving_average[i,objectID] = v
            distances = []
            distances_2 = []
            for n in range(len(xs)):
                d = calculateDistance(centroid[0], centroid[1], xs[n], ys[n])
                distances.append(d)
                distances_2.append(calculateDistance(centroid[0], v, xs[n], ys[n]))
                
            min_d = min(distances)
            dd.append(min_d)
            dd_2.append(min(distances_2))
            
#             locc = distances.index(min_d)
#             print(min_d)
#         print("##################################################")
total_error = int(sum(dd))
avg_error = int(total_error/len(dd))
print("Total distance error for Centroid Tracker is: "+str(total_error))
print("Average distance error for Centroid Tracker is: "+str(avg_error))

total_error = int(sum(dd_2))
avg_error = int(total_error/len(dd_2))
print("Total distance error for Centroid tracker integrated with moving average is: "+str(total_error))
print("Average distance error for Centroid tracker integrated with moving average is: "+str(avg_error))


# # ARIMA Centroid Tracker

# In[16]:


yy_arima_centroid = np.zeros(shape = (len(images), max_objects), dtype = int)

for i in range(max_objects):
    yyy = YY[:,i]
    model = ARIMA(yyy, order=(1,0,0))
    model_fit = model.fit()
    output = model_fit.predict()
    yy_arima_centroid[:,i] = output

for k in range (len(images)):
    img = images[k].copy()
    db = data[i]
    xs = db['x']
    ys = db['y']
    idss = db['id']
    for j in range(yy_arima_centroid.shape[1]):
        if (int(XX[k,j]) !=0 and int(yy_arima_centroid[k,j]) !=0):
            distances = []
            for n in range(len(xs)):
                d = calculateDistance(XX[k,j], yy_arima_centroid[k,j], xs[n], ys[n])
                distances.append(d)

            min_d = min(distances)
            dd.append(min_d)
            
total_error = int(sum(dd))
avg_error = int(total_error/len(dd))
print("Total distance error for Centroid Tracker integrated with ARIMA model is: "+str(total_error))
print("Average distance error for Centroid Tracker integrated with ARIMA model is: "+str(avg_error))


# # SORT Tracker utilizing Kalman filter

# In[17]:


global frame_count
global tracker_list
global max_age
global min_hits
global track_id_list
global debug

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter
max_age = maxDisappeared

min_hits = 1  # no. of consecutive matches needed to establish a track

tracker_list = [] # list for trackers
# list for track ID (0-49)
# track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'])
track_id_list= deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])

debug = True

xx_kalman = np.zeros(shape=(len(images), max_objects))
yy_kalman = np.zeros(shape=(len(images), max_objects))
length_path = len(images)
labelled = []

if len(pred_car) == len(pred_bus):
    print("Initializing Kalman filter based tracker")
    dd = []

    for i, car in enumerate(pred_car):
        z_box = car + pred_bus[i]
        frame_count+=1
        img = images[i].copy()
        x_box =[]
        db = data[i]
        xs = db['x']
        ys = db['y']
        idss = db['id']        

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
                distances = []
                for n in range(len(xs)):
                    d = calculateDistance(center_x, center_y, xs[n], ys[n])
                    distances.append(d)

                min_d = min(distances)
                dd.append(min_d)
                
        deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  

        for trk in deleted_tracks:
            track_id_list.append(trk.id)

        tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

cv2.destroyAllWindows()
total_error = int(sum(dd))
avg_error = int(total_error/len(dd))
print("Total distance error for SORT Tracker is: "+str(total_error))
print("Average distance error for SORT Tracker is: "+str(avg_error))


# # Trace trackers on each frame

# In[18]:


# Red color for Centroid, Yellow for Centroid+Moving Average, Blue for centroid + ARIMA, Black for ground and Green for Kalman Filter
multiple_colors = True

trace_ground = False
trace_centroid = True
trace_moving_average = False
trace_arima_centroid = False
trace_sort = False

centroid_color = (0, 0, 255)            # Red color
moving_average_color = (0, 255, 255)    # Yellow color
arima_centroid_color = (255, 0, 0)      # Blue color
ground_color = (0, 0, 0)                # Black color
sort_color = (0, 255, 0)                # Green color
thickness = 1                           # Thickness of trajectory line
distance_threshold = 30

length_path = len(images)
frames = copy.deepcopy(images)

def trace(x_matrix, y_matrix, color):
    pts_list = []
    colors = []
    for i in range(max_objects):
        pts_list.append(deque(maxlen=length_path))
        if multiple_colors:
            colors.append((np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)))
    
    for k in range(len(images)):
        for j in range(x_matrix.shape[1]):
            pts_list[j].appendleft((int(x_matrix[k,j]), int(y_matrix[k,j])))
            
            for i in range(1, len(pts_list[j])):
                if pts_list[j][i - 1] is None or pts_list[j][i] is None:
                    continue
                
                d = calculateDistance(pts_list[j][i - 1][0], pts_list[j][i-1][1], pts_list[j][i][0], pts_list[j][i][1])
                if d<distance_threshold:
                    if multiple_colors:
                        cv2.line(frames[k], pts_list[j][i - 1], pts_list[j][i], colors[j], thickness) 
                    else:
                        cv2.line(frames[k], pts_list[j][i - 1], pts_list[j][i], color, thickness)                    

if trace_ground:
    trace(xx_ground, yy_ground, ground_color)
if trace_centroid:
    trace(XX, YY, centroid_color)
if trace_moving_average:
    trace(XX, yy_moving_average, moving_average_color)
if trace_arima_centroid:
    trace(XX, yy_arima_centroid, arima_centroid_color)
if trace_sort:
    trace(xx_kalman, yy_kalman, sort_color)
                
for i in range (len(frames)):
    cv2.imshow("img", frames[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()
        

cv2.destroyAllWindows()

# In[19]:


output_video_path = os.path.join(os.getcwd(), "centroid_tracker_mask.mp4")
convert_frames_to_video(frames, output_video_path)


# In[ ]:




