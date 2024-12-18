from ultralytics import YOLO
import cv2
import time

# 設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target = 'city.mp4'  # =0相機 , 'city.mp4'
model = YOLO('yolov8x.pt')  # n,s,m,l,x 五種大小

names = model.names
print(names)

cap = cv2.VideoCapture(target)

while True:
    st = time.time()  
    r, frame = cap.read()
    if not r:
        break

    results = model(frame, verbose=False)  # 用AI模型分析 ,verbose=False不顯示訊息

    car_count = 0  # 計算車子數量變數
    bus_count = 0  # 計算巴士數量變數
    truck_count = 0  # 計算卡車數量變數

    # 分析每一個box，包含6個欄位
    for box in results[0].boxes.data:
        x1 = int(box[0])  # 左
        y1 = int(box[1])  # 上
        x2 = int(box[2])  # 右
        y2 = int(box[3])  # 下
        r = round(float(box[4]), 2)  # 信任度
        n = names[int(box[5])]  # 物件名字

        # 檢查是否為car, bus, truck
        if n == 'car':
            # 畫框 bounding box，影像 左上 右下 顏色 粗細
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 自行畫框
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # 自行標物件名稱
            car_count += 1  # 計算車子數量
        elif n == 'bus':
            # 畫框 bounding box，影像 左上 右下 顏色 粗細
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 綠色框
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # 自行標物件名稱
            bus_count += 1  # 計算巴士數量
        elif n == 'truck':
            # 畫框 bounding box，影像 左上 右下 顏色 粗細
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 藍色框
            cv2.putText(frame, n, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # 自行標物件名稱
            truck_count += 1  # 計算卡車數量

    # 顯示車輛總數
    total_count = car_count + bus_count + truck_count
    cv2.putText(frame, f'Total vehicles = {total_count} (Cars: {car_count}, Buses: {bus_count}, Trucks: {truck_count})',
                (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)

    # 計算FPS
    et = time.time()
    FPS = round((1 / (et - st)), 1)  # 評估時間

    cv2.putText(frame, f'FPS = {FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('YOLOv8', frame)

    key = cv2.waitKey(1)
    if key == 27:  # 按ESC退出
        break
