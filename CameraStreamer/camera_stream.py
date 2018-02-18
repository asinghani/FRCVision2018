import cv2
import cscore as cs

class CameraStream:
    cvSource = None

    def __init__(self, resolution = (1280, 720), framerate = 30, port = 5800):
        self.cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG,
                                    resolution[0], resolution[1], framerate)
        mjpegServer = cs.MjpegServer("cvhttpserver", port)
        mjpegServer.setSource(self.cvSource)

    def putFrame(self, frame):
        self.cvSource.putFrame(frame)
