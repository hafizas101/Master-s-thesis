# Trajectory estimation of vehicles in crowded and crossroad scenarios
This repository holds the code and results for my Master's thesis available on [ResearchGate](https://www.researchgate.net/publication/346963904_Deep_Learning-based_Trajectory_Estimation_of_Vehicles_in_Crowded_and_Crossroad_Scenarios). The video demo of this project is in results directory and on [YouTube](https://www.youtube.com/channel/UC-F03u7wrjzrhDZBfjK4-8w?view_as=subscriber) . In the below figure, black represents ground truth trajectory, green represents SORT tracker based trajectory and red represents centroid tracker based trajectory.
<p align="center">
  <img width="600" height="320" src="https://github.com/hafizas101/Master-s-thesis/blob/master/combined.png">
</p>

## Detection and Localization Code
It contains test portion of UA-DETRAC dataset data and annotation folders according to weather categories defined by the authors of this dataset. Download weights file from [Google Drive](https://drive.google.com/file/d/18Y2f61mW0sq4jHBAjaYkgDNf-uMa4nfV/view?usp=sharing) and place it in /yolo_files/ directory.
The files corresponding to this part are 'yolov3_process.ipynb', 'detector.py' and 'detector_help.py'.
- yolov3_process.ipynb: Jupyter notebook to run detection and measure Average Precision (AP) for a weather portion.
- detector.py: Input the method and weather which you want to process. The running command for this file is:
~~~
python detector.py --method yolo --weather cloudy
~~~
In order to process other weathers, you can replace cloudy with night, rainy or sunny. You can replace yolo with mask but you should have Mask RCNN benchmark installed on your machine and you should be in demo directory of Mask RCNN Benchmark. I recommend to start work with yolo. 
- detector_help.py: Helping file that defines some functions, constants and classes being used in detector.py.

## Tracking of vehicles Code
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
Since this is a big dataset and it takes almost 7 or 8 hours to process a single weather portion, so I have saved the predicted bounding boxes in json file. So you can get detection metrics using this file in a few seconds or minutes. First download and extract 'results.zip' file from [Google Drive](https://drive.google.com/file/d/1iP-nl0mQOOpnARCqz7YLpTArHpbPz440/view?usp=sharing) and place all files in results directory. The running command for this file is:
~~~
python detect.py --method yolo --weather cloudy
~~~
You can replace yolo with mask and cloudy with night, rainy or sunny.
