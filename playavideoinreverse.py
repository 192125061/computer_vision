import cv2

cap = cv2.VideoCapture("E:/computer_vision/test.mp4")
ret, vid = cap.read()
count = 0
ret = True
frame_list = []
while(ret == True):
    ret, vid = cap.read()
    frame_list.append(vid)
    count += 1
frame_list.pop()
for frame in frame_list:
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) and 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
frame_list.reverse()
for frame in frame_list:
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) and 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
