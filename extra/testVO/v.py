import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(320.0, 180.0, 963.93956241, 966.19092668, 644.61135581, 356.78176629)
vo = VisualOdometry(cam)

traj = np.zeros((600,600,3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

i = 0

#cv2.namedWindow("x")
cv2.namedWindow("map")

ret, rgb_img = cap.read()
print(rgb_img.shape)

while True:
	i = i + 1

	ret, rgb_img = cap.read()
	img = cv2.cvtColor(cv2.resize(rgb_img, (320, 180)), cv2.COLOR_BGR2GRAY)

	vo.update(img)

	if (i > 2):
		x, y, z = vo.cur_t[0], vo.cur_t[1], vo.cur_t[2]
	else:
		x, y, z = 0., 0., 0.

	print("Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z))

	# cv2.imwrite('map.png', traj)

	draw_x, draw_y = (int(x)*5)+300, (int(y)*5) + 300

	cv2.circle(traj, (draw_x,draw_y), 2, (0, 255, 0), 3)

	#cv2.imshow('x', rgb_img)
	cv2.imshow('map', traj)
	cv2.waitKey(1)
