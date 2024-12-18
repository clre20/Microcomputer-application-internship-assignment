from ultralytics import YOLO
import cv2
import time

# 設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

target = 'Counter.mp4'  # 目標影片名稱
model = YOLO('yolov8x.pt')  # 載入YOLOv8模型，選擇合適的大小 (n,s,m,l,x)

names = model.names  # 物件名稱列表
print(names)

cap = cv2.VideoCapture(target)  # 開啟影片檔案

while True:
    st = time.time()  # 計算FPS
    r, frame = cap.read()  # 讀取每一幀
    if not r:
        break  # 如果沒有讀取到影片畫面，退出循環

    results = model(frame, verbose=False)  # 用AI模型分析影像，verbose=False表示不顯示訊息

    person_count = 0  # 計算人物數量

    # 分析每個預測框（box）
    for box in results[0].boxes.data:
        x1 = int(box[0])  # 左邊界
        y1 = int(box[1])  # 上邊界
        x2 = int(box[2])  # 右邊界
        y2 = int(box[3])  # 下邊界
        confidence = round(float(box[4]), 2)  # 信任度
        label = names[int(box[5])]  # 物件名稱

        # 檢查物件是否為人物 ('person')
        if label == 'person':
            # 畫出bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 用紅色畫框
            # 顯示物件名稱及信任度
            cv2.putText(frame, f'{label} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)  # 顯示物件名稱和信任度
            person_count += 1  # 計算人物數量

    # 顯示總人物數量
    cv2.putText(frame, f'Total persons = {person_count}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv2.LINE_AA)

    # 計算FPS
    et = time.time()
    FPS = round((1 / (et - st)), 1)  # 每幀處理時間，反推FPS

    cv2.putText(frame, f'FPS = {FPS}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    
    # 顯示處理過的影像
    cv2.imshow('YOLOv8', frame)

    key = cv2.waitKey(1)  # 等待按鍵，1ms後繼續執行
    if key == 27:  # 按下ESC退出
        break

cap.release()  # 釋放資源
cv2.destroyAllWindows()  # 關閉所有OpenCV視窗
