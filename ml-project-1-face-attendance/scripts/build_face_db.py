import os
import pickle
import face_recognition

def build_face_database(db_path):
    """
    构建人脸特征库
    :param db_path: 人脸照片文件夹路径（每个子文件夹对应一个人，文件夹名就是人名）
    """
    known_encodings = []
    known_names = []

    # 遍历人脸库中的每个子文件夹（对应一个人）
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"正在处理 {person_name} 的照片...")
        # 遍历该人文件夹下的所有照片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                # 加载图片并提取人脸特征
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:  # 如果图片中检测到人脸
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
            except Exception as e:
                print(f"跳过错误图片 {img_path}: {e}")

    # 保存人脸特征到文件
    output_path = os.path.join(db_path, "encodings.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    
    print(f"✅ 人脸库构建完成，已保存到 {output_path}")

if __name__ == "__main__":
    # 这里改成你的人脸照片文件夹路径
    FACE_DB_PATH = r"D:\ml-project-1-face-attendance\face_db"
    build_face_database(FACE_DB_PATH)