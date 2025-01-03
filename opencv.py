import cv2

# โหลดโมเดล
model = cv2.dnn_DetectionModel('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# โหลดคลาสเลเบล
with open('labels.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# กำหนดแหล่งวิดีโอ (เว็บแคม)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('ไม่สามารถเปิดเว็บแคมได้')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ทำการตรวจจับวัตถุ
    classIds, confs, bbox = model.detect(frame, confThreshold=0.5)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 1:  # classId 1 หมายถึง 'person' ใน COCO dataset
                cv2.rectangle(frame, box, (0, 255, 0), 2)
                cv2.putText(frame, f'{classLabels[classId-1]}: {confidence:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # แสดงผลเฟรมที่ได้
    cv2.imshow('Frame', frame)
    
    # กด 'q' เพื่อออกจากวิดีโอ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
