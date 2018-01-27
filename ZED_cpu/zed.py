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
        left, right = numpy.vsplit(frame, 2)
