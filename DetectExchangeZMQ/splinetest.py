import math, cv2
import numpy as np
import pathfinder as pf

points = [
    pf.Waypoint(-2, 5, math.radians(90 + 30)),
    pf.Waypoint(0, 0, math.radians(90))
]

img = np.zeros((500, 500, 3))

info, trajectory = pf.generate(points, pf.FIT_HERMITE_CUBIC, pf.SAMPLES_HIGH,
                               dt=0.05, # 50ms
                               max_velocity=5.0,
                               max_acceleration=15.0,
                               max_jerk=60.0)

modifier = pf.modifiers.TankModifier(trajectory).modify(2.33) # TODO MEASURE WHEELBASE WIDTH
left = modifier.getLeftTrajectory()
right = modifier.getRightTrajectory()


for segment in left:
	print(segment.x, segment.y)
	cv2.circle(img, ((img.shape[1] // 2) - int(segment.x * 50), int(segment.y * 50)), 3, (255, 255, 0), -1)


for segment in right:
        print(segment.x, segment.y)
        cv2.circle(img, ((img.shape[1] // 2) - int(segment.x * 50), int(segment.y * 50)), 3, (0, 255, 255), -1)


for segment in trajectory:
        print(segment.x, segment.y)
        cv2.circle(img, ((img.shape[1] // 2) - int(segment.x * 50), int(segment.y * 50)), 3, (255, 255, 255), -1)


cv2.imshow("path",img)
cv2.waitKey(0)
