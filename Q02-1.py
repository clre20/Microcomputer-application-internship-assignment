from ultralytics import YOLO
import cv2
import time
import numpy as np
from shapely.geometry import Polygon

# 設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

# 設定影片路徑
target = 'Counter.mp4'
model = YOLO('yolov8n.pt')  # 預設模型：n,s,m,l,x 五種大小

names = model.names  # 認識的80物件字典：編號及名稱
print(names)

# 區域三維陣列
area = [
    [[1017, 38], [1089, 842], [1425, 830], [1173, 34]],  # 區域1 (紅色)
    [[1427, 58], [1679, 55], [1912, 552], [1912, 812], [1818, 824], [1812, 715]],  # 區域2 (綠色)
    [[754, 227], [707, 325], [694, 568], [655, 784], [330, 782], [414, 583], [475, 461], [518, 360], [616, 212]]  # 新增區域3 (藍色)
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
    b = [[object[0], object[1]], [object[2], object[1]], [object[2], object[3]], [object[0], object[3]]]  # 取得人的四個角組成poly物件
    for i in range(len(area)):
        poly1 = Polygon(b)  # 人的多邊形
        poly2 = Polygon(area[i])  # 第i區的多邊形
        intersection_area = poly1.intersection(poly2).area  # 重疊區域部分多少畫素
        poly1Area = poly1.area  # 人區域一共有多少畫素
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

    personCount = [0, 0, 0]  # 初始化人數 (區域1, 區域2, 區域3)
    for box in results[0].boxes.data:
        x1 = int(box[0])  # 左
        y1 = int(box[1])  # 上
        x2 = int(box[2])  # 右
        y2 = int(box[3])  # 下
        r = round(float(box[4]), 2)  # 信任度
        n = names[int(box[5])]  # 物體名稱
        if n != 'person':  # 只關注人
            continue

        tempObj = [x1, y1, x2, y2, r, n]
        ObjInArea = inarea(tempObj, area)  # 計算物件在不同區域比例陣列
        
        # 區域1 紅色 (比例>=25%)
        if ObjInArea[0] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            personCount[0] += 1  # 區域1人數+1

        # 區域2 綠色 (比例>=25%)
        if ObjInArea[1] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            personCount[1] += 1  # 區域2人數+1
        
        # 區域3 藍色 (比例>=25%)
        if ObjInArea[2] >= 25:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            personCount[2] += 1  # 區域3人數+1

    # 印出每個區域的人數
    print(f"Area0: {personCount[0]} people")
    print(f"Area1: {personCount[1]} people")
    print(f"Area2: {personCount[2]} people")

    # 顯示區域0、區域1、區域3的人數
    cv2.putText(frame, f'Area0={personCount[0]}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area1={personCount[1]}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Area2={personCount[2]}', (20, 140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)

    # 計算FPS
    et = time.time()
    FPS = round(1 / (et - st), 1)
    cv2.putText(frame, f'FPS={FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)

    # 顯示畫面
    cv2.imshow('YOLOv8', frame)

    # 按Esc鍵退出
    key = cv2.waitKey(1)
    if key == 27:  # 按下ESC退出
        break

cap.release()
cv2.destroyAllWindows()
