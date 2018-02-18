import cv2, time, os, glob

for img in glob.glob("./img*.jpg"):
    os.remove(img)

cap = cv2.VideoCapture(1)
i = 0

while not (raw_input("Press enter to continue or 'q' to exit: ") == "q"):
    ret, frame = cap.read()

    cv2.imshow("Image", frame)
    cv2.imwrite("img"+str(i)+".jpg", frame)
    cv2.waitKey(2)
    i = i + 1