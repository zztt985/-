import cv2
import face_recognition
import sqlite3
import numpy as np

# 连接到数据库
conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()

# 加载已存储的人脸编码
cursor.execute("SELECT name, face_encoding FROM users")
users_data = cursor.fetchall()
stored_encodings = []
user_names = []

for user_data in users_data:
    name, encoding = user_data
    stored_encodings.append(np.frombuffer(encoding, dtype=np.float64))
    user_names.append(name)

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为RGB格式
    rgb_frame = frame[:, :, ::-1]

    # 检测人脸
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 对比识别的脸与已存储的人脸编码
        matches = face_recognition.compare_faces(stored_encodings, face_encoding)

        if True in matches:
            match_index = matches.index(True)
            name = user_names[match_index]
            cv2.putText(frame, f"Welcome {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示摄像头画面
    cv2.imshow('Face Recognition - Login', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
