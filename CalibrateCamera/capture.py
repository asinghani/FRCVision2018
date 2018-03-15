import cv2, time, os, glob

for img in glob.glob("./img*.jpg"):
    os.remove(img)

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
i = 0

while not (input("Press enter to continue or 'q' to exit: ") == "q"):
    ret, frame = cap.read()

    cv2.imshow("Image", cv2.resize(frame, (640, 360)))
    cv2.imwrite("img"+str(i)+".jpg", frame)
    cv2.waitKey(2)
    i = i + 1
