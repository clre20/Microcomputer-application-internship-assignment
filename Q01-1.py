from ultralytics import YOLO
import cv2
import time

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target = 'city.mp4'
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
    car_count = 0
    bus_count = 0 
    truck_count = 0 

    # 分析每一個box，包含6個欄位
    for box in results[0].boxes.data:
        x1 = int(box[0])
        y1 = int(box[1])  
        x2 = int(box[2]) 
        y2 = int(box[3])
        r = round(float(box[4]), 2) 
        n = names[int(box[5])]  

        if n == 'car':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  
            car_count += 1  
        elif n == 'bus':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  
            bus_count += 1  
        elif n == 'truck':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  
            truck_count += 1  

    total_count = car_count + bus_count + truck_count
    cv2.putText(frame, f'Total vehicles = {total_count} (Cars: {car_count}, Buses: {bus_count}, Trucks: {truck_count})',
                (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)

    et = time.time()
    FPS = round((1 / (et - st)), 1)  

    cv2.putText(frame, f'FPS = {FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
