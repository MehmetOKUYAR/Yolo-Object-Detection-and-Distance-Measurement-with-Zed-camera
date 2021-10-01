# Yolo Object Detection and Distance Measurement with Zed camera

you can test object detection and distance measurement with zed camera


### How to use 

You need to run this script like that `python zed.py `
or if you use tensorRT yolo, You need to run this script like that `python zed_trt.py `
You need to edit the codes in `zed.py` line according to yourself.

specify the yolo weights and config files you trained before.
~~~~~~~~~~~~
7. weightsPath = "yolov4-obj_last.weights"
8. configPath = "yolov4-obj.cfg"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the video path you want to test.

~~~~~~~~~~
14. cap = cv2.VideoCapture('2.mp4')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

then you need to edit the following lines in `zed.py` 

Size should be changed according to your config file
~~~~~~
8. model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)
~~~~~~~~~~~~~~~~~~~~
Edit them according to your class labels.
~~~~~~~~~~~~
13.  LABELS = [ 'class_name1',
                'class_name2',
                'class_name3',
                'class_name3',
                .
                .
                .
                ]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
that's all, if your model predicts well enough, it will predict and start labeling.
After the process is finished, you can check the labels with a labeling program and edit them again.
## You can see how the program works in the gif below.

![into mp4](https://github.com/MehmetOKUYAR/Zed_Yolo_distance_measurement/blob/master/intro.mp4)

