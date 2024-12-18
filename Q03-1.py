from ultralytics import YOLO
import cv2,time

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)
target=0
model = YOLO('CLYU.pt')
names=model.names
print(names)
cap=cv2.VideoCapture(target)
while 1:
    st=time.time()  
    r,frame = cap.read()
    if r==False:
        break
    results = model(frame,verbose = False)
    frame= results[0].plot()
    et=time.time() 
    FPS=int(1/(et-st))
    cv2.putText(frame, 'FPS=' + str(FPS), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5)
    cv2.imshow('YOLOv8', frame)
    key=cv2.waitKey(1)
    if key==27:
        break

