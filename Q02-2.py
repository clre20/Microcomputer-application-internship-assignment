from ultralytics import YOLO
import cv2
import time
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)
target = 'city.mp4'
model = YOLO('yolov8n.pt')
names = model.names
print(names)
area = [

    [[425, 13], [536, 339], [376, 1071], [39, 1033], [3, 669], [281, 289], [371, 17]],
    [[465, 2], [592, 117], [732, 445], [673, 1077], [388, 1076], [546, 336], [440, 17]],
    [[701, 1071], [721, 810], [777, 678], [867, 538], [1002, 409], [1103, 336], [1372, 249], [1392, 288], [1324, 336], [1279, 378], [1211, 431], [1181, 476], [1139, 568], [1091, 712], [1024, 925], [996, 1060], [852, 1048], [797, 1060]],
]

# 繪製區域
def drawArea(f, area, color, th):
    for a in area:
        v = np.array(a, np.int32)
        cv2.polylines(f, [v], isClosed=True, color=color, thickness=th)
    return f

# 取得重疊比例
def inarea(object, area):
    inAreaPercent = [] 
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]]  # 取得車輛的四個角組成poly物件
    for i in range(len(area)):
        poly1 = Polygon(b) 
        poly2 = Polygon(area[i]) 
        poly1 = make_valid(poly1) 
        poly2 = make_valid(poly2)        
        try:
            intersection_area = poly1.intersection(poly2).area  
        except Exception as e:
            intersection_area = 0  
            print(f"Error in intersection calculation: {e}")
        
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

    vehicleCount = [0, 0, 0]  
    for box in results[0].boxes.data:
        x1 = int(box[0])  
        y1 = int(box[1])  
        x2 = int(box[2])  
        y2 = int(box[3])  
        r = round(float(box[4]), 2)  
        n = names[int(box[5])] 
        if n != 'car':  
            continue

        tempObj = [x1, y1, x2, y2, r, n]
        ObjInArea = inarea(tempObj, area) 
        
        if ObjInArea[0] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            vehicleCount[0] += 1  
    
        if ObjInArea[1] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            vehicleCount[1] += 1  
      
        if ObjInArea[2] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            vehicleCount[2] += 1 
            
    print(f"Area0: {vehicleCount[0]} vehicles")
    print(f"Area1: {vehicleCount[1]} vehicles")
    print(f"Area2: {vehicleCount[2]} vehicles")
    cv2.putText(frame, f'Area0={vehicleCount[0]}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area1={vehicleCount[1]}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area2={vehicleCount[2]}', (20, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)
    et = time.time()
    print(f"Processing Time: {et - st:.2f}s")
    
    cv2.imshow("YOLOv8", frame)
    # 按Esc鍵退出
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
