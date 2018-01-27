import socket
import cv2
import numpy as np

s = socket.socket()
s.bind(("127.0.0.1", 5050))

s.listen(1)
c, addr = s.accept()

print("Connection from: " + str(addr))

packet = None

while True:
    length = int(c.recv(24).decode().replace(" ", ""))
    #print(length)

    packet = c.recv(1024)
    while len(packet) < length:
        packet = packet + c.recv(1024)
        #print(len(packet))

    #print(len(packet))

    buffer = np.frombuffer(packet, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    print(image.shape)
    cv2.imshow("Image", image)


    print("sending: " + str("r"))
    c.send("r".encode())

    cv2.waitKey(1)

c.close()
