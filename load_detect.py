import numpy as np
import time, glob, os, natsort, sys, cv2, json, argparse
from detector_help import *


parser = argparse.ArgumentParser(description='UA-DETRAC Dataset analysis using YOLOv3 or Mask RCNN Benchmark')
parser.add_argument('--weather', type=str,   help='Write name of weather (cloudy, night, rainy or sunny) for UA-DETRAC')
parser.add_argument('--method', type=str,    help='Write name of method (yolo or maskrcnn) , default:yolo ')
args = parser.parse_args()
weather = args.weather
method = args.method

xml_paths = glob.glob(os.path.join(os.getcwd(), "weather_data/"+str(weather)+"_annotations/*"))
sorted_xml_paths = natsort.natsorted(xml_paths, reverse=False)
folder_dirs = glob.glob(os.path.join(os.getcwd(), "weather_data/"+str(weather)+"_data/*"))
sorted_folder_dirs = natsort.natsorted(folder_dirs, reverse=False)

iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

if method == "yolo":
	with open('results/yolov3_car_'+str(weather)+'_pred_dict.json') as json_file:
	    dict_pred_car = json.load(json_file)
	with open('results/yolov3_bus_'+str(weather)+'_pred_dict.json') as json_file:
	    dict_pred_bus = json.load(json_file)

elif method == "mask":
	with open('results/mask_car_'+str(weather)+'_pred_dict.json') as json_file:
	    dict_pred_car = json.load(json_file)
	with open('results/mask_bus_'+str(weather)+'_pred_dict.json') as json_file:
	    dict_pred_bus = json.load(json_file)


import xml.etree.ElementTree as ET

img_id = []
ground_car = []
ground_bus = []
total_ground_car = 0
total_ground_bus = 0
for k, folder in enumerate(sorted_xml_paths):
    tree = ET.parse(folder)
    root = tree.getroot()
    frames = tree.findall('frame')
    path = sorted_folder_dirs[k]+"/*.jpg"
    file_paths = glob.glob(path)
    num = len(file_paths)

    if len(frames)!=num:
        print("Number of images and their annotations are not same. You may need to remove some images whose annotations are not provided in UA-DETRAC annotation")
    else:
        for i in range (num):
            gg = frames[i].attrib
            img_id.append(i+1)
            ground_boxes = frames[i].findall('target_list/target')
            locations_car = []
            locations_bus = []
            for j, child in enumerate(ground_boxes):
                box = child.findall('box')
                attribute = child.findall('attribute')
                aa = box[0].attrib
                x1 = int(float(aa.get('left')))
                y1 = int(float(aa.get('top')))
                w = int(float(aa.get('width')))
                h = int(float(aa.get('height')))
                x2 = x1+w
                y2 = y1+h

                bb = attribute[0].attrib
                label = bb.get('vehicle_type')

                if label == "car":
                    locations_car.append([x1, y1, x2, y2])
                elif label == "bus":
                    locations_bus.append([x1, y1, x2, y2])

            total_ground_car = total_ground_car + len(locations_car)
            dictionary_car = {'locations': locations_car}

            total_ground_bus = total_ground_bus + len(locations_bus)
            dictionary_bus = {'locations': locations_bus}

            ground_car.append(dictionary_car)
            ground_bus.append(dictionary_bus)


dict_ground_car = {'image_id': img_id, 'frame_ground': ground_car}
dict_ground_bus = {'image_id': img_id, 'frame_ground': ground_bus}


# # Analysis for cars

# In[ ]:


precision_list_list = []
recall_list_list = []
for k in range (len(iou_thresholds)):
    iou_threshold = iou_thresholds[k]
    print(iou_threshold)
    all_iou = []
    tp_list = []
    fp_list = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    ground_no_predict_no = 0
    ground_yes_predict_no = 0
    ground_no_predict_yes = 0
    ground_yes_predict_yes = 0
#     confidences = []
    precision_list = []
    recall_list = []
    tpr_list = []
    fpr_list = []

    IDs_ground = dict_ground_car['image_id'] # List of integers
    IDs_pred = dict_pred_car['image_id']

    p_ground = dict_ground_car['frame_ground'] # List of dictionaries
    p_pred = dict_pred_car['frame_predictions']

    if len(p_ground) != len(p_pred):
        print("Number of images in prediction phase and ground truth are not same")
    else:
        for l, pp in enumerate (p_pred): # Iterate through all images.
            pp_ground = p_ground[l]
            pp_pred = pp
            
            ground_truth = pp_ground['locations']
            loc = pp_pred['locations']
            ious = []
            predictions = []
            c1 = len(loc)
            c2 = len(ground_truth)
            
            if c1==0 and c2==0:
                tn = tn + 1
                ground_no_predict_no = ground_no_predict_no + 1
                continue
            elif c1==0 and c2 !=0:
                ground_yes_predict_no = ground_yes_predict_no + 1
                fn = fn + 1
                continue
            elif c1!=0 and c2==0:
                ground_no_predict_yes = ground_no_predict_yes + 1
                fp = fp + 1
                continue
# The the purpose of below nested loop is to look for the match of ground box and predicted box.
# Inner loop looks for the match of a single ground box.
# Outer loop looks for all predicting boxes of a particular frame.
            for i in range (c1): # Iterate through a single image
                dis = []
                for j in range (c2):
                    d = bb_intersection_over_union(loc[i], ground_truth[j])
                    dis.append(d)
#                 print(dis)

                iou_box = max(dis) # At maximum value, there is overlap in ground and predicted box
                ious.append(iou_box)
                index = dis.index(iou_box)
#                 confidences.append(conf_pred[index])
                if iou_box > iou_threshold:
                    tp_list.append(1)
                    fp_list.append(0)
                    tp = tp+1
                else:
                    tp_list.append(0)
                    fp_list.append(1)
                    fp = fp + 1
                precision = tp/(tp+fp)
                recall = tp/total_ground_car
                precision_list.append(precision)
                recall_list.append(recall)             
                iou_img = sum(ious)/len(ious)
                all_iou.append(iou_img)
                if tn+fp != 0:
                    fpr_list.append(fp/(tn+fp))
                    tpr_list.append(precision)
        if len(all_iou) != 0:        
	        avg_iou = sum(all_iou)/len(all_iou)
	        print("Avergae IoU for all images: {}".format(avg_iou))

        AP = CalculateAveragePrecision(recall_list, precision_list)
        print("Average Precision: "+str(AP))

        precision_list_list.append(precision_list)
        recall_list_list.append(recall_list)


# In[ ]:


for i in range (len(recall_list_list)):
    plt.plot(recall_list_list[i], precision_list_list[i], label = str(iou_thresholds[i]))


plt.xlim(0,1)
plt.ylim(0,1)
plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


# # Analysis for Bus

# In[ ]:


precision_list_list = []
recall_list_list = []
for k in range (len(iou_thresholds)):
    iou_threshold = iou_thresholds[k]
    print(iou_threshold)
    all_iou = []
    tp_list = []
    fp_list = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    ground_no_predict_no = 0
    ground_yes_predict_no = 0
    ground_no_predict_yes = 0
    ground_yes_predict_yes = 0
#     confidences = []
    precision_list = []
    recall_list = []
    tpr_list = []
    fpr_list = []

    IDs_ground = dict_ground_bus['image_id'] # List of integers
    IDs_pred = dict_pred_bus['image_id']

    p_ground = dict_ground_bus['frame_ground'] # List of dictionaries
    p_pred = dict_pred_bus['frame_predictions']

    if len(p_ground) != len(p_pred):
        print("Number of images in prediction phase and ground truth are not same")
    else:
        for l, pp in enumerate (p_pred): # Iterate through all images.
            pp_ground = p_ground[l]
            pp_pred = pp
            
            ground_truth = pp_ground['locations']
            loc = pp_pred['locations']
            ious = []
            predictions = []
            c1 = len(loc)
            c2 = len(ground_truth)

            if c1==0 and c2==0:
                tn = tn + 1
                ground_no_predict_no = ground_no_predict_no + 1
                continue
            elif c1==0 and c2 !=0:
                ground_yes_predict_no = ground_yes_predict_no + 1
                fn = fn + 1
                continue
            elif c1!=0 and c2==0:
                ground_no_predict_yes = ground_no_predict_yes + 1
                fp = fp + 1
                continue
# The the purpose of below nested loop is to look for the match of ground box and predicted box.
# Inner loop looks for the match of a single ground box.
# Outer loop looks for all predicting boxes of a particular frame.
            for i in range (c1): # Iterate through a single image
                dis = []
                for j in range (c2):
                    d = bb_intersection_over_union(loc[i], ground_truth[j])
                    dis.append(d)
#                 print(dis)

                iou_box = max(dis) # At maximum value, there is overlap in ground and predicted box
                ious.append(iou_box)
                index = dis.index(iou_box)
#                 confidences.append(conf_pred[index])
                if iou_box > iou_threshold:
                    tp_list.append(1)
                    fp_list.append(0)
                    tp = tp+1
                else:
                    tp_list.append(0)
                    fp_list.append(1)
                    fp = fp + 1
                precision = tp/(tp+fp)
                recall = tp/total_ground_bus
                precision_list.append(precision)
                recall_list.append(recall)             
                iou_img = sum(ious)/len(ious)
                all_iou.append(iou_img)
                if tn+fp != 0:
                    fpr_list.append(fp/(tn+fp))
                    tpr_list.append(precision)
                
        if len(all_iou) != 0:        
	        avg_iou = sum(all_iou)/len(all_iou)
	        print("Avergae IoU for all images: {}".format(avg_iou))

        AP = CalculateAveragePrecision(recall_list, precision_list)
        print("Average Precision: "+str(AP))

        precision_list_list.append(precision_list)
        recall_list_list.append(recall_list)


# In[ ]:


for i in range (len(recall_list_list)):
    plt.plot(recall_list_list[i], precision_list_list[i], label = str(iou_thresholds[i]))

plt.xlim(0,1)
plt.ylim(0,1)
plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
