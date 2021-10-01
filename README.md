# Yolo Object Detection and Distance Measurement with Zed camera

you can test object detection and distance measurement with zed camera


### How to use 

You need to run this script like that `python zed.py `
or if you use tensorRT yolo, You need to run this script like that `python zed_trt.py `
You need to edit the codes in `zed.py` line according to yourself.

specify the yolo weights and config files you trained before.
~~~~~~~~~~~~
45. weightsPath = "yolov4-obj_last.weights"
46. configPath = "yolov4-obj.cfg"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

then you need to edit the following lines in `zed.py` 

Size should be changed according to your config file
~~~~~~
56. model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)
~~~~~~~~~~~~~~~~~~~~
Edit them according to your class labels.
~~~~~~~~~~~~
68.  LABELS = [ 'class_name1',
                'class_name2',
                'class_name3',
                'class_name3',
                .
                .
                .
                ]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
that's all, if you have a zed camera you can easily find the distance of the objects you have detected
## You can see how the program works in the gif below.

![into gif](https://github.com/MehmetOKUYAR/Zed_Yolo_distance_measurement/blob/master/intro.gif)

