#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import socket
import numpy as np

image_pub = rospy.Publisher("image_topic_2", Image)
rospy.init_node('image_publisher', anonymous=True)

s = socket.socket()
s.bind(("0.0.0.0", 5050))

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
        print(len(packet))

    #print(len(packet))

    buffer = np.frombuffer(packet, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    print(len(packet))
    #print(image.shape)
    #cv2.imshow("Image", image)

    msg_frame = CvBridge().cv2_to_imgmsg(image, "bgr8")
    image_pub.publish(msg_frame)
    cv2.waitKey(100)

    print("sending: " + str("r"))
    c.send("r".encode())

    cv2.waitKey(1)

c.close()
