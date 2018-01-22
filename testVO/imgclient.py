import socket
import cv2

print(cv2.__version__)

s = socket.socket()
s.connect(("54.183.207.181", 5051))

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    data = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)), [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1]
    print(len(data))

    s.send(str(len(data)).ljust(24).encode())
    s.send(data)
    output = s.recv(1024).decode()

    print('Received from server: ' + output)


s.close()