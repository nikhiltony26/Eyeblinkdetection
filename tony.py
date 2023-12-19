# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import threading
import os
import ctypes
import sys
import logging
from logging.handlers import RotatingFileHandler
import requests
import queue

# Set the telegram chat id and bot token
try:
    TELEGRAM_CHAT_ID = os.getenv('1781637279')
    TELEGRAM_BOT_TOKEN = os.getenv('6397644141:AAEFf9xek5T8LwlRClgWIGGigI6xChsDtdY')
    USE_TELEGRAM = True
except KeyError:
    USE_TELEGRAM = False

# Create a queue to hold the jobs
telegram_queue = queue.Queue()

# Set up logging configuration
LOG_LEVEL = logging.DEBUG
LOG_FILENAME = "blink_detector.log"
LOG_MAX_BYTES = 10 * 1024 * 1024 # 10mb
LOG_BACKUP_COUNT = 2
# create a logger
logger = logging.getLogger('')
logger.setLevel(LOG_LEVEL)
# create a file handler which logs even debug messages
fh = RotatingFileHandler(LOG_FILENAME, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
fh.setLevel(LOG_LEVEL)

# create a console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

# create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :%(levelname)s:%(funcName)s:%(lineno)d %(message)s', datefmt='%d-%b-%y %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def send_telegram_photo(chat_id, photo, bot_token, message):
    url = f"https://api.telegram.org/bot6397644141:AAEFf9xek5T8LwlRClgWIGGigI6xChsDtdY/sendPhoto?chat_id=1781637279&caption={message}"
    try:
        with open(photo, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': chat_id,
                    'caption': message}
            response = requests.post(url, files=files)
            response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred when sending telegram photo: {e}")
    finally:
        os.remove(photo)

def send_telegram_photo_thread(chat_id, photo, bot_token, message):
    telegram_queue.put((chat_id, photo, bot_token, message))


def send_telegram_photo_worker():
    while True:
        job = telegram_queue.get()
        if job is None:
            break
        chat_id, photo, bot_token, message = job
        send_telegram_photo(chat_id, photo, bot_token, message)
        telegram_queue.task_done()


def play_alarm(stop_event):
    # play an alarm sound and keep playing it until the stop event is set
    while not stop_event.is_set():
        logger.info("playing alarm.wav")
        if sys.platform.startswith('win'):
            # Play the default sound for Windows
            ctypes.windll.user32.MessageBeep(-1)
            time.sleep(1)
        elif sys.platform.startswith('darwin'):
            # Play the default sound for macOS
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            time.sleep(0.5)
        elif sys.platform.startswith('linux'):
            # Play the default sound for Linux
            os.system("aplay /usr/share/sounds/gnome/default/alerts/glass.ogg")
            time.sleep(0.5)
        else:
            logger.error("Unsupported platform")
            raise ValueError("Unsupported platform")


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.27,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=5,
	help="the number of consecutive frames the eye must be below the threshold")

def main():
    args = vars(ap.parse_args())
    EYE_AR_THRESH = args['threshold']
    EYE_AR_CONSEC_FRAMES = args['frames']
    PROGRAM_ENABLE = True
    ALARM_ON = False
    ALARM_THREAD = None
    ALARM_STOP_EVENT = threading.Event()    

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    BLINK_TIMESTAMPS = []

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    logger.debug("loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    logger.info("starting video stream thread...")
    logger.info("print q to quit...")

    if args['video'] == "camera":
        vs = VideoStream(src=0).start()
        fileStream = False
    else:
        vs = FileVideoStream(args["video"]).start()
        fileStream = True
   
    time.sleep(1.0)
    
    # loop over frames from the video stream
    logger.debug("starting main loop...")

    # Start the worker thread
    logger.debug("starting telegram worker thread...")
    telegram_thread = threading.Thread(target=send_telegram_photo_worker)
    telegram_thread.start()
    while True:
    	# if this is a file video stream, then we need to check if
    	# there any more frames left in the buffer to process
        if fileStream and not vs.more():
              break    
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale
    	# channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	# detect faces in the grayscale frame   
        rects = detector(gray, 0)
        
    
    	# loop over the face detections
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:
    			# if the eyes were closed for a sufficient number of
    			# then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES and PROGRAM_ENABLE:
                    TOTAL += 1
                    BLINK_TIMESTAMPS.append(time.time())
                    logger.debug(f"blink detected! blinks in the past 5s: {len(BLINK_TIMESTAMPS)}")
    
    			# reset the eye frame counter
                COUNTER = 0

            BLINK_TIMESTAMPS = [t for t in BLINK_TIMESTAMPS if time.time() - t <= 5]

    		# draw the total number of blinks on the frame along with
    		# the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks (last 5s): {}".format(len(BLINK_TIMESTAMPS)), (10, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Alarm: {}".format(ALARM_ON), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Program Enabled: {}".format(PROGRAM_ENABLE), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # function to trigger alarm when user blinks more than 5 times in 5 seconds and alarm is not already on
            if len(BLINK_TIMESTAMPS) >= 5 and not ALARM_ON and PROGRAM_ENABLE:
                logger.debug("alarm triggered!")
                ALARM_ON = True
                ALARM_STOP_EVENT.clear()
                ALARM_THREAD = threading.Thread(target=play_alarm, args=(ALARM_STOP_EVENT,))
                ALARM_THREAD.start()
                BLINK_TIMESTAMPS = []
                TOTAL = 0
                if USE_TELEGRAM:
                    cv2.imwrite("frame.jpg", frame)
                    message = "Alarm triggered at {0}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    send_telegram_photo_thread(TELEGRAM_CHAT_ID, "frame.jpg", TELEGRAM_BOT_TOKEN, message)
            # if the alarm is on and the user blinks 5 times in 5 seconds, turn off the alarm
            # killing the thread running the alarm, and reset the blink timestamps and total
            if len(BLINK_TIMESTAMPS) >= 5 and ALARM_ON:
                 logger.debug("alarm stopped!")
                 ALARM_ON = False
                 ALARM_STOP_EVENT.set()
                 if ALARM_THREAD is not None:
                     ALARM_THREAD.join()
                 BLINK_TIMESTAMPS = []
                 TOTAL = 0
                 if USE_TELEGRAM:
                    cv2.imwrite("frame.jpg", frame)
                    message = "Alarm stopped at {0}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    send_telegram_photo_thread(TELEGRAM_CHAT_ID, "frame.jpg", TELEGRAM_BOT_TOKEN, message)
     
    	# show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
     
    	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            if USE_TELEGRAM:
                    cv2.imwrite("frame.jpg", frame)
                    message = "q pressed at {0}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    send_telegram_photo_thread(TELEGRAM_CHAT_ID, "frame.jpg", TELEGRAM_BOT_TOKEN, message)
            break
        
        # if the `s` key was pressed, toggle program enable
        if key == ord("s"):
            PROGRAM_ENABLE = not PROGRAM_ENABLE
            logger.info(f"s key pressed. program enable: {PROGRAM_ENABLE}")
            if USE_TELEGRAM:
                    cv2.imwrite("frame.jpg", frame)
                    message = "q pressed at {0}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    send_telegram_photo_thread(TELEGRAM_CHAT_ID, "frame.jpg", TELEGRAM_BOT_TOKEN, message)
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    telegram_queue.put(None)
    telegram_thread.join()


if __name__ == '__main__' :
    main()