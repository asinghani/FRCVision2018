import cv2, numpy
from threading import Thread

class ZEDWrapper:
    camera = None
    def __init__(self, cameraNumber = 1, resolution = (1280, 720),
                 framerate = 30, leftDistortCoeff = None, rightDistortCoeff = None):
        self.camera = cv2.VideoCapture(1)
        self.camera.set(3, 2 * resolution[0])
        self.camera.set(4, resolution[1])
        self.camera.set(5, framerate)

        self.left = None
        self.right = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            ret, frame = self.camera.read()
            self.left, self.right = numpy.hsplit(frame, 2)

            if self.stopped:
                self.camera.close()
                return

