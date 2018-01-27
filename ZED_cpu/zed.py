import cv2, numpy

class ZEDWrapper:
    camera = None
    def __init__(self, cameraNumber = 1, resolution = (1280, 720),
                 framerate = 30, leftDistortCoeff = None, rightDistortCoeff = None):
        self.camera = cv2.VideoCapture(1)
        self.camera.set(3, 2 * resolution[0])
        self.camera.set(4, resolution[1])
        self.camera.set(5, framerate)

    def getFrame(self):
        ret, frame = self.camera.read()
        left, right = numpy.hsplit(frame, 2)
        return (left, right)

if __name__ == "__main__":
    zed = ZEDWrapper()
    while True:
        left, right = zed.getFrame()
        cv2.imshow("img", left)
        cv2.waitKey(1)
