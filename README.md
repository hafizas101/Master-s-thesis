# Trajectory estimation of vehicles in crowded and crossroad scenarios
This repository holds the code and results for my Master's thesis available on https://www.overleaf.com/7453397376mnbqbhttzczd
The video demo of this project is in results directory. We propose tracking by detection based trajectory estimation pipeline which consists of two stages: The first stage is the detection and localization of vehicles and the second stage is Kalman Filter based tracking. We analyze the performance of **Mask RCNN benchmark** and **YOLOv3** on **UA DETRAC dataset** which is a large scale real life traffic dataset. We evaluate certain metrics like inference time, Intersection over union (IoU), Precision Recall (PR) curve and mean Average Precision. Experiments show that Mask RCNN benchmark outperforms YOLOv3 in speed and accuracy as compared to YOLOv3. After the detection, we analyze the performance of **centroid tracker** and **kalman filter** based tracker. Experiments show that Kalman filter based tracking gives us much more smooth and accurate trajectory as compared to noisy trajectory obtained from centroid tracker.

## Detection and Localization
Download and extract 'weather_data.zip' file from https:. It contains test portion of UA-DETRAC dataset data and annotation folders according to weather categories defined by the authors of this dataset.
The files corresponding to this part are 'yolov3_process.ipynb', 'detector.py' and 'detector_help.py'.
- yolov3_process.ipynb: Jupyter notebook to run detection and measure Average Precision (AP) for a weather portion.
- detector.py: Input the method and weather which you want to process. The running command for this file is:
~~~
python detector.py --method yolo --weather cloudy
~~~
In order to process other weathers, you can replace cloudy with night, rainy or sunny. You can replace yolo with mask but you should have Mask RCNN benchmark installed on your machine and you should be in demo directory of Mask RCNN Benchmark. I recommend to start work with yolo. 
- detector_help.py: Helping file that defines some functions, constants and classes being used in detector.py.

## Tracking of vehicles
In order to run a short demo of tracking, we are using some images in MVI_40852 test portion because this is a crowded and crossroad scenario. The results are in results directory.
The files corresponding to this part are 'tracking_yolov3.ipynb', 'tracker.py' and 'tracker_help.py'.
- tracking_yolov3.ipynb: Jupyter notebook to run Centroid and Kalman Filter based tracking.
- tracker.py: Input the method which you want to use for detection part. The running command for this file is:
~~~
python tracker.py --method yolo
~~~
You can replace yolo with mask but you should have Mask RCNN benchmark installed on your machine and you should be in demo directory of Mask RCNN Benchmark. I recommend to start work with yolo. 
- tracker_help.py: Helping file that defines some functions, constants and classes being used in tracker.py.

## load_detect.py
Since this is a big dataset and it takes almost 7 or 8 hours to process a single weather portion, so I have saved the predicted bounding boxes in json file. So you can get detection metrics using this file in a few seconds or minutes. First download and extract 'results.zip' file from https://drive.google.com/file/d/1iP-nl0mQOOpnARCqz7YLpTArHpbPz440/view?usp=sharing The running command for this file is:
~~~
python detect.py --method yolo --weather cloudy
~~~
You can replace yolo with mask and cloudy with night, rainy or sunny.
