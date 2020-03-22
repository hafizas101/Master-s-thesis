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
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


labelsPath = os.path.join(os.getcwd(), "yolo_files/classes.names")
weightsPath = os.path.join(os.getcwd(), "yolo_files/yolov3.weights")
configPath = os.path.join(os.getcwd(), "yolo_files/yolov3.cfg")
confidence_t = 0.5
threshold_t = 0.3
img_size = 416


cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml")
cfg.merge_from_list([])
cfg.freeze()
# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(cfg,confidence_threshold=0.7,show_mask_heatmaps=False,masks_per_dim=2,min_image_size=800)


maxDisappeared = 10  # no.of consecutive unmatched detection before a track is deleted
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.8
font_color = (255, 0, 0)
output_FPS = 10  # Frames per second of output video
max_objects = 50 # Maximum number of vehicles that can be in a frame at some time. Should not exceed this value.


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
	    # print(coco_demo.CATEGORIES[labels[i]])
	    if coco_demo.CATEGORIES[labels[i]] == "car":
	        box = box.to(torch.int64)
	        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
	        location = top_left + bottom_right
	        s = scores[i].tolist()

	        locations_car.append(location)
	        confi_car.append(s)
	        cv2.rectangle(image, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]),(0, 255, 0), 2)
	        cv2.rectangle(image, (top_left[0]-1, top_left[1]-15), (bottom_right[0]+1, top_left[1]), (0,255,0), -1, 1)
	        text = coco_demo.CATEGORIES[labels[i]]
	        cv2.putText(image,text,(top_left[0],top_left[1]-2), font, font_size, font_color, 1, cv2.LINE_AA)


	    elif coco_demo.CATEGORIES[labels[i]] == "bus":
	        box = box.to(torch.int64)
	        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
	        location = top_left + bottom_right
	        s = scores[i].tolist()

	        locations_bus.append(location)
	        confi_bus.append(s)
	        cv2.rectangle(image, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]),(0, 255, 0), 2)
	        cv2.rectangle(image, (top_left[0]-1, top_left[1]-15), (bottom_right[0]+1, top_left[1]), (0,255,0), -1, 1)
	        text = coco_demo.CATEGORIES[labels[i]]
	        cv2.putText(image,text,(top_left[0],top_left[1]-2), font, font_size, font_color, 1, cv2.LINE_AA)

	return locations_car, locations_bus, image




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
            l = np.array([x,y,x+w, y+h])
#             print((LABELS[classIDs[i]]))

            # draw a bounding box rectangle and label on the image
            if (LABELS[classIDs[i]] == "car"):
                confi_car.append(confidences[i])
                locations_car.append(l)
                cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
                cv2.rectangle(image, (x-1, y-15), (x+w+1, y), (0,255,0), -1, 1)
                # Output the labels that show the x and y coordinates of the bounding box center.
                text = LABELS[classIDs[i]]
                cv2.putText(image,text,(x,y-3), font, font_size, font_color, 1, cv2.LINE_AA)
                
            elif (LABELS[classIDs[i]] == "bus"):
                confi_bus.append(confidences[i])
                locations_bus.append(l)
                cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
                cv2.rectangle(image, (x-1, y-15), (x+w+1, y), (0,255,0), -1, 1)
                # Output the labels that show the x and y coordinates of the bounding box center.
                text = LABELS[classIDs[i]]
                cv2.putText(img,text,(x,y-3), font, font_size, font_color, 1, cv2.LINE_AA)

    return locations_car, locations_bus, image




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








# # Kalman Filter based tracking

# In[21]:


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
        Implment only the predict stage. This is used for unmatched detections and 
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)



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
