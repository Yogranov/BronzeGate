BronzeGate is an automatic gate system for PalGate app users, the system scan license plate and send an API call to PalGate services if the number exists in your list.

The main algorithm base on YoloV4, the AI was training on 4000 different images, specific to the Israeli license plates style, and reach a precision of more than 90%.

Main features:
    1. Single / Dual mode - allow to open the gate from both sides.
    2. Motion detection - the AI will start to work just if movement has detected.
    3. Debug mode
    4. Extra lite - works on tiny model, can be run on Raspberry pie or Jetson nano.


Main configurations:
```
allowPlateSize = 0.2                        # in percentage, ratio to the entire frame
waitingTime = 2                             # sleep the program (in seconds) when plate was recognized
timeToDetect = 120                          # the time in seconds the ai will run after movement detected
motionDetectionThreshold = 150000           # how sensitive to be before start the AI
numberFile = script_dir + 'numbers.txt'     # list of all valid license plates
videoWidth = 1200                           #
videoHeight = 700                           #
clearBuffer = 10                            # number of frames to clear after detected number, fix double open gate cause camera buffer
singleCamMode = False                       #true = single camera, false dual camera
debug = True                                # will print info to console and won't open the gate
```

Gif from debug section:
<img src="https://github.com/Yogranov/BronzeGate/blob/master/README_MEDIA/debug-anim.gif" width="600" height="313" />



External links:
    The project forked from 'the ai guy'- https://github.com/theAIGuysCode/yolov4-custom-functions
    How to get PalGate tokens by Roei Ofri - https://github.com/RoeiOfri/homebridge-palgate-opener
    Plate Recognition website - https://platerecognizer.com/