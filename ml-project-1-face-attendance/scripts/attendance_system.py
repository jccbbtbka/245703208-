import cv2
import pickle
import datetime
import os
import face_recognition

# -------------------------- 配置 --------------------------
FACE_ENCODINGS_PATH = r"D:\ml-project-1-face-attendance\face_db\encodings.pkl"
ATTENDANCE_LOG_PATH = r"D:\ml-project-1-face-attendance\attendance_log"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TOLERANCE = 0.45

# -------------------------- 初始化 --------------------------
if not os.path.exists(ATTENDANCE_LOG_PATH):
    os.makedirs(ATTENDANCE_LOG_PATH)

print("正在加载人脸库...")
try:
    with open(FACE_ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
    print("✅ 人脸库加载完成！")
except FileNotFoundError:
    print("❌ 请先运行 build_face_db.py")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

attended_today = set()
today_date = datetime.date.today().strftime("%Y-%m-%d")
log_file = os.path.join(ATTENDANCE_LOG_PATH, f"attendance_{today_date}.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("姓名,签到时间\n")

# -------------------------- 主程序（正确识别逻辑） --------------------------
print("📷 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # ✅ 正确识别：取距离最近 + 严格阈值
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_idx = distances.argmin()
        name = "Unknown"

        if distances[best_idx] < TOLERANCE:
            name = known_names[best_idx]

        # 签到
        if name != "Unknown" and name not in attended_today:
            attended_today.add(name)
            now = datetime.datetime.now().strftime("%H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{name},{now}\n")
            print(f"✅ {name} 签到成功：{now}")

        # 还原位置
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 画框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()