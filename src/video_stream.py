import cv2
from threading import Thread

class WebcamStream:

    def __init__(self, src=0):
        # CAP_DSHOW to help webcam stable in windows
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        # keep buffer size =1 to get latest frame
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                continue
            self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
