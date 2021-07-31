
import time
import cv2
import tensorflow as tf
import requests

# GPU configs 

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# config = tf.compat.v1.ConfigProto
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config)
# tf.config.gpu_options.set_per_process_memory_fraction(0.4)
# 
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import io
from sys import platform

if platform == 'win32':
    script_dir = 'C:\path\to\project/' # windows
else:
    script_dir = '/home/user/path/to/project' # linux

# general settings
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


# video sources - can be 0,1.. for live feed, or video path for testing
video_src1 = 0
video_src2 = 1


#palgate settings
palgateUserToken = ''
gateID = ''


#plate recognizer API Token
platerecognizerToken = ''


# YoloV4 settings
iou = 0.45
score = 0.70
tiny = True
model = 'yolov4'
input_size = 416
model = script_dir + 'checkpoints/yolov4-tiny-416'


# load YoloV4 models to ram
saved_model_loaded = tf.saved_model.load(model, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


# load plates numbers from file
allowNumbers = open(numberFile, 'r')
allowNumbers = allowNumbers.readlines()
for i in range(len(allowNumbers)):
    allowNumbers[i] = allowNumbers[i].replace('\n', '')



def main():
    print('Progrem started')
    runAi = False

    vid = cv2.VideoCapture(video_src1)
    if not singleCamMode:
        vid2 = cv2.VideoCapture(video_src2)

    ret, frame = vid.read()
    ret, frame2 = vid.read()
    frame = cv2.resize(frame, (videoWidth, videoHeight)) 
    frame2 = cv2.resize(frame2, (videoWidth, videoHeight))
    
    if not singleCamMode:
        secRet, secFrame = vid2.read()
        secRet, secFrame2 = vid2.read()
        secFrame = cv2.resize(secFrame, (videoWidth, videoHeight)) 
        secFrame2 = cv2.resize(secFrame2, (videoWidth, videoHeight))

    
    turn = 1
    currentFrame = None
    while True:
        numberDetected = False
        if debug:
            start_time = time.time()

        if ret and (singleCamMode or secRet):
            
            if turn == 1:
                currentFrame = frame
                currentFrame2 = frame2
            else:
                currentFrame = secFrame
                currentFrame2 = secFrame2
            
            if not runAi and detectMovevent(currentFrame, currentFrame2):
                aiRunTime = int(time.time())
                runAi = True
                
            if runAi:
                numberDetected = runAI(currentFrame)
                if int(time.time()) > aiRunTime + timeToDetect:
                    runAi = False
            

            if numberDetected:
                clearLoop = clearBuffer
            else:
                clearLoop = 1

            for _ in range(clearLoop):
                
                if debug:
                    cv2.imshow('frame', frame)
                frame = frame2
                ret, frame2 = vid.read()
                frame2 = cv2.resize(frame2, (videoWidth, videoHeight))
               
                if not singleCamMode:
                    if debug:
                        cv2.imshow('secFrame', secFrame)
                    secFrame = secFrame2
                    secRet, secFrame2 = vid2.read()
                    secFrame2 = cv2.resize(secFrame2, (videoWidth, videoHeight))
            

            if not singleCamMode:
                turn = turn * -1


            if debug:
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            vid.release()
            cv2.destroyAllWindows()
            break

def runAI(frame):
    numberDetected = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)


    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]


    detected = False
    if pred_bbox['tf_op_layer_concat_14'][0].numpy().any():
        detected = True


    if detected:
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        
        try:
            boxes, scores, classes, num_objects = pred_bbox
            xmin, ymin, xmax, ymax = boxes[0]
            cropped_img = frame[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        except:
            pass
            

        if cropped_img.any() and cropped_img.shape[1] > int(frame.shape[1] * allowPlateSize):
            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            plateNumber = readViaWeb(cropped_img)

            if plateNumber != None:
                if checkPlate(plateNumber):
                    openGate()
            numberDetected = True
                


            time.sleep(waitingTime)
            if debug:
                cv2.imshow('Cropped Plate', cropped_img)

    return numberDetected

def detectMovevent(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    diff = cv2.dilate(diff, None, iterations = 3)

    if platform == 'win32':
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < motionDetectionThreshold:
            continue
        
        print("Movement Detected!")
        return True
    
    return False


def readViaWeb(img):
    im = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    if not debug:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=['il']),
            files=dict(upload=rawBytes),
            headers={'Authorization': 'Token ' + platerecognizerToken}
        )
    number = None
    try:
        if response.json()['results']:
            number = response.json()['results'][0]['plate']
    except:
        pass
    
    return number


def openGate():
    print('opening gate')
    headers = {
        'X-Requested-With': 'XMLHttpRequest',
        'x-bt-user-token': palgateUserToken
    }
    if not debug:
        requests.get('https://api1.pal-es.com/v1/bt/device/' + gateID + '/open-gate?outputNum=1', headers=headers)

    print('opened')


def checkPlate(number):
    if number in allowNumbers:
        return True
    
    return False

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



#### the program starts from here ###
if __name__ == '__main__':
    main()

exit(0)
    