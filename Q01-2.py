from ultralytics import YOLO
import cv2
import time

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target = 'Counter.mp4'
model = YOLO('yolov8x.pt')

names = model.names 
print(names)

cap = cv2.VideoCapture(target) 

while True:
    st = time.time()  
    r, frame = cap.read()  
    if not r:
        break  

    results = model(frame, verbose=False)  

    person_count = 0 

    # 分析每個預測框（box）
    for box in results[0].boxes.data:
        x1 = int(box[0])  
        y1 = int(box[1])  
        x2 = int(box[2])  
        y2 = int(box[3])  
        confidence = round(float(box[4]), 2)  
        label = names[int(box[5])] 

        if label == 'person':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f'{label} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)  
            person_count += 1 
            
    cv2.putText(frame, f'Total persons = {person_count}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)
    et = time.time()
    FPS = round((1 / (et - st)), 1) 
    cv2.putText(frame, f'FPS = {FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
