import numpy as np
import cv2
import time, math
from threadedstreamer import WebcamVideoStream
from imutils.video import FPS
import imutils

img1 = cv2.imread('data/finaltarget.png',0)
#cap = zed.ZEDWrapper(resolution = (960, 540), framerate = 10)
#cap.start()
cap2 = WebcamVideoStream().start() #cv2.VideoCapture(0)
#cap2.stream.set(3, 1280)
#cap2.stream.set(4, 720)
fps = FPS().start()

#cap2.set(5, 30)
#cap2.set(cv2.CAP_PROP_FRAME_COUNT, 1)

cameraMatrix = np.array([np.array([1.43134719e+03 / 1.5, 0.00000000e+00, 9.88360958e+02 / 1.5]),
                         np.array([0.00000000e+00, 1.43704194e+03 / 1.5, 5.40208244e+02 / 1.5]),
                         np.array([0.00000000e+00, 0.00000000e+00, 1.00000000e+00])])

cameraDistortionCoefficients = \
    np.array([0.07022623, -0.31233022, -0.00558322, -0.00174155, 0.25068144])

objectPoints = np.array([np.array([0.000, 4.375, 0.000]),
                         np.array([0.000, 0.000, 0.000]),
                         np.array([2.375, 0.000, 0.000]),
                         np.array([2.375, 4.375, 0.000])])



current_milli_time = lambda: int(round(time.time() * 1000))


#sift = cv2.ORB_create()
sift = cv2.xfeatures2d.SURF_create()

t = current_milli_time()
kp1 = sift.detect(img1,None)
kp1, des1 = sift.compute(img1,kp1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


i = ""

cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("img3", 400, 200)

def baseround(x, base=5):
    return int(base * round(float(x)/base))

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R[0])
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    R = cv2.Rodrigues(R)[0]
    #print(R)
    #print (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x * 180.0 / 3.1415926, y * 180.0 / 3.1415926, z * 180.0 / 3.1415926])

def roundArray(A, base=1):
    return np.array([baseround(A[0], base), baseround(A[1], base), baseround(A[2], base)])

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    #left, right = cap.getFrame()
    frame = cap2.read()
    #frame = cap.left
    #if frame == None:
    #    continue

    img2 = cv2.undistort(frame, cameraMatrix, cameraDistortionCoefficients)

    t = current_milli_time()
    kp2, des2 = sift.detectAndCompute(img2,None)
    print("SIFT time: {}".format(current_milli_time() - t))

    t = current_milli_time()

    # Remove np.asarray to use with SIFT
    #matches = flann.knnMatch(np.asarray(des1, np.float32),np.asarray(des2, np.float32),k=2)
    matches = flann.knnMatch(des1, des2, k=2)

    print("FLANN time: {}".format(current_milli_time() - t))


    t = current_milli_time()
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #print(M)
        if (not (M is None)) and M.shape[0] != None:
            dst = cv2.perspectiveTransform(pts,M)
        else:
            dst = []
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #print(dst)
        try:
            img2 = cv2.circle(img2, (dst[0][0][0], dst[0][0][1]), 30, (0, 255, 0), -1) # Top Left
            img2 = cv2.circle(img2, (dst[1][0][0], dst[1][0][1]), 30, (0, 255, 255), -1) # Bottom Left
            img2 = cv2.circle(img2, (dst[2][0][0], dst[2][0][1]), 30, (0, 0, 255), -1) # Bottom Right
            img2 = cv2.circle(img2, (dst[3][0][0], dst[3][0][1]), 30, (255, 255, 0), -1) # Top Right

            pnp = cv2.solvePnPRansac(objectPoints, dst, cameraMatrix, cameraDistortionCoefficients)
            print(pnp[1])
            print(roundArray(rotationMatrixToEulerAngles(pnp[1])))
            #eulerAngles = rotationMatrixToEulerAngles(cv2.Rodrigues(pnp[1]))
            #print(eulerAngles)

            print("X = {}\nY = {}\nZ = {}\nXt = {}\nYt = {}\nZt = {}"
                  .format(pnp[1][0][0], pnp[1][1][0], pnp[1][2][0]))
        except Exception as err:
            print(err)


    else:
        print("Not enough matches are found - %d/%d".format(len(good),10))
        matchesMask = None


    draw_params = dict(matchColor = (255,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    print("Homography Time time: {}".format(current_milli_time() - t))


    cv2.imshow("img3", cv2.resize(img3, (0, 0), fx = 0.5, fy = 0.5))
    #cv2.imwrite("output"+i+".png", img3)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    #fps.update()
    #print(fps.fps())
