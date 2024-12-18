from ultralytics import YOLO
import cv2
import time
import numpy as np
from shapely.geometry import Polygon

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

# 設定影片路徑
target = 'Counter.mp4'
model = YOLO('yolov8n.pt') 

names = model.names  
print(names)
area = [
    [[1017, 38], [1089, 842], [1425, 830], [1173, 34]],  
    [[1427, 58], [1679, 55], [1912, 552], [1912, 812], [1818, 824], [1812, 715]],  
    [[754, 227], [707, 325], [694, 568], [655, 784], [330, 782], [414, 583], [475, 461], [518, 360], [616, 212]]
]

def drawArea(f, area, color, th):
    for a in area:
        v = np.array(a, np.int32)
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)
    return f

def inarea(object, area):
    inAreaPercent = []
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]] 
    for i in range(len(area)):
        poly1 = Polygon(b)  
        poly2 = Polygon(area[i])  
        intersection_area = poly1.intersection(poly2).area  
        poly1Area = poly1.area  
        overlap_percent = (intersection_area / poly1Area) * 100  
        inAreaPercent.append(overlap_percent)  
    return inAreaPercent

cap = cv2.VideoCapture(target)

while True:
    st = time.time()
    r, frame = cap.read()
    if not r: 
        break

    results = model(frame, verbose=False) 
    frame = drawArea(frame, [area[0]], (0, 0, 255), 3) 
    frame = drawArea(frame, [area[1]], (0, 255, 0), 3) 
    frame = drawArea(frame, [area[2]], (255, 0, 0), 3)  

    personCount = [0, 0, 0]  
    for box in results[0].boxes.data:
        x1 = int(box[0])  
        y1 = int(box[1]) 
        x2 = int(box[2])  
        y2 = int(box[3]) 
        r = round(float(box[4]), 2)  
        n = names[int(box[5])]  
        if n != 'person':  
            continue

        tempObj = [x1, y1, x2, y2, r, n]
        ObjInArea = inarea(tempObj, area) 
        
        if ObjInArea[0] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            personCount[0] += 1 
            
        if ObjInArea[1] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            personCount[1] += 1 
        
        if ObjInArea[2] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            personCount[2] += 1 

    print(f"Area0: {personCount[0]} people")
    print(f"Area1: {personCount[1]} people")
    print(f"Area2: {personCount[2]} people")

    cv2.putText(frame, f'Area0={personCount[0]}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area1={personCount[1]}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area2={personCount[2]}', (20, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)

    et = time.time()
    FPS = round(1 / (et - st), 1)
    cv2.putText(frame, f'FPS={FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)
    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
