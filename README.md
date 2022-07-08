# Yolo Object Detection and Distance Measurement with Zed camera

you can test object detection and distance measurement with zed camera


### How to use 
You need to run this script like that `python zed.py `
If you use tensorRT yolo, You need to run this script like that `python zed_trt.py `
You need to edit the codes in `zed.py` line according to yourself.

The default values for weight, config, names file and ZED camera ID are
~~~~~~~~~~~~
    config_path = "yolov4-tiny.cfg"
    weight_path = "yolov4-tiny.weights"
    meta_path = "coco.names"
    svo_path = None
    zed_id = 0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Download the model file, for instance Yolov4-Tiny
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

### Making changes in the "zed.py" file
You need to edit the following lines in `zed.py` 

Size should be changed according to image height and width value in .cfg file
Default values are - width : 608, height : 608
~~~~~~
98. model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)
~~~~~~~~~~~~~~~~~~~~
### Run the application
To launch the ZED with YOLO simply run the script :

        python3 zed.py

The input parameters can be changed using the command line :

        python3 zed.py -c <config> -w <weight> -m <meta> -s <svo_file> -z <zed_id>

For instance :

        python3 zed.py -c yolov4-tiny.cfg -w yolov4-tiny.weights -m coco.names -z 1
        
For running with custom weights :

        python3 zed.py -c yolov4-custom.cfg -w yolov4-custom.weights -m obj.names -z 1

To display the help :

        python3 zed.py -h
 
that's all, if you have a zed camera you can easily find the distance of the objects you have detected
## You can see how the program works in the gif below.


<p align="center">
  <img src="intro.gif" width=100%>
</p>
