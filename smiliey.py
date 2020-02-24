from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from datetime import datetime
from imutils import face_utils
import imutils
import time
import dlib
import cv2

def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar


COUNTER = 0
TOTAL = 0


shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

fps= FPS().start()
cv2.namedWindow("Frame",cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    # frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth= shape[mStart:mEnd]
        mar= smile(mouth)
        mouthHull = cv2.convexHull(mouth)
        #print(shape)
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "Say Cheeeze!".format(mar), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)

        if mar <= .35 or mar > .38 :
            COUNTER += 1
        else:
            if COUNTER >= 3:
                TOTAL += 1
                frame = vs.read()
                time.sleep(.1)
                frame2= frame.copy()
                cv2.putText(frame, "Looking Good!".format(mar), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 1)
                cv2.putText(frame, "Hit d to delete".format(mar), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Frame",frame)
                key3 = cv2.waitKey(0) & 0xFF
                if key3 != ord('d'):
                    img_name = "images/opencv_frame_{}.png".format(datetime.timestamp(datetime.now()))
                    cv2.imwrite(img_name, frame2)
                    time.sleep(.3)
                    print("{} written!".format(img_name))
            COUNTER = 0


        # cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    fps.update()

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break

fps.stop()
cv2.destroyAllWindows()
vs.stop()
