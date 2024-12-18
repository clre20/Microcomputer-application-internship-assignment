from ultralytics import YOLO
import cv2
import time
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid  # 新增：import make_valid

# 設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

# 設定影片路徑
target = 'city.mp4'
model = YOLO('yolov8n.pt')  # 預設模型：n,s,m,l,x 五種大小

names = model.names  # 認識的80物件字典：編號及名稱
print(names)

# 區域三維陣列 (假設你已經知道這些區域的頂點座標)
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
    inAreaPercent = []  # area陣列，物件在所有區域的比例
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]]  # 取得車輛的四個角組成poly物件
    for i in range(len(area)):
        poly1 = Polygon(b)  # 車輛的多邊形
        poly2 = Polygon(area[i])  # 第i區的多邊形
        poly1 = make_valid(poly1)  # 修正無效多邊形
        poly2 = make_valid(poly2)  # 修正無效多邊形
        
        try:
            intersection_area = poly1.intersection(poly2).area  # 重疊區域部分多少畫素
        except Exception as e:
            intersection_area = 0  # 如果出現錯誤，將交集面積設為0
            print(f"Error in intersection calculation: {e}")
        
        poly1Area = poly1.area  # 車輛區域一共有多少畫素
        overlap_percent = (intersection_area / poly1Area) * 100  # 在區域中的比例
        inAreaPercent.append(overlap_percent)  # 加入比例陣列
    return inAreaPercent

cap = cv2.VideoCapture(target)

while True:
    st = time.time()
    r, frame = cap.read()
    if not r:  # 讀取失敗
        break

    results = model(frame, verbose=False)  # YOLO辨識 verbose=False不顯示文字結果
    frame = drawArea(frame, [area[0]], (0, 0, 255), 3)  # 繪製區域1 (紅色)
    frame = drawArea(frame, [area[1]], (0, 255, 0), 3)  # 繪製區域2 (綠色)
    frame = drawArea(frame, [area[2]], (255, 0, 0), 3)  # 繪製區域3 (藍色)

    vehicleCount = [0, 0, 0]  # 初始化車輛數量 (區域1, 區域2, 區域3)
    for box in results[0].boxes.data:
        x1 = int(box[0])  # 左
        y1 = int(box[1])  # 上
        x2 = int(box[2])  # 右
        y2 = int(box[3])  # 下
        r = round(float(box[4]), 2)  # 信任度
        n = names[int(box[5])]  # 物體名稱
        if n != 'car':  # 只關注車輛
            continue

        tempObj = [x1, y1, x2, y2, r, n]
        ObjInArea = inarea(tempObj, area)  # 計算物件在不同區域比例陣列
        
        # 區域1 紅色 (比例>=25%)
        if ObjInArea[0] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            vehicleCount[0] += 1  # 區域1車輛數量+1

        # 區域2 綠色 (比例>=25%)
        if ObjInArea[1] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            vehicleCount[1] += 1  # 區域2車輛數量+1
        
        # 區域3 藍色 (比例>=25%)
        if ObjInArea[2] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            vehicleCount[2] += 1  # 區域3車輛數量+1

    # 印出每個區域的車輛數量
    print(f"Area0: {vehicleCount[0]} vehicles")
    print(f"Area1: {vehicleCount[1]} vehicles")
    print(f"Area2: {vehicleCount[2]} vehicles")

    # 顯示區域0、區域1、區域3的車輛數量
    cv2.putText(frame, f'Area0={vehicleCount[0]}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area1={vehicleCount[1]}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area2={vehicleCount[2]}', (20, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)

    # 顯示每幀處理時間
    et = time.time()
    print(f"Processing Time: {et - st:.2f}s")
    
    cv2.imshow("YOLOv8", frame)
    # 按Esc鍵退出
    key = cv2.waitKey(1)
    if key == 27:  # 按下ESC退出
        break


cap.release()
cv2.destroyAllWindows()
