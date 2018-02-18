import cv2
import socket
import pickle
import struct

cap = cv2.VideoCapture(0)
#client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket.connect(("localhost", 8089))

while True:
    ret, frame = cap.read()
    img_str = cv2.imencode('.bmp', cv2.resize(frame, (640, 480)))[1].tostring()
    print(len(img_str))

    data = pickle.dumps( )
    clocket.sendall(struct.pack("H", len(data))+data)ient_s