import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
import numpy as np
import pickle
import time


# --------------------- 全局配置 ---------------------
FACE_DATA_DIR = "face_data"
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.pickle"
USER_DB_PATH = "user_db.pickle"  # 新增：用户数据库文件路径
USER_DB = {}  # {username: (password, face_id, last_login_method)}
FACE_ID_COUNTER = 0

# 初始化人脸识别模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 手机 IP 摄像头地址（替换为你的实际地址！）
PHONE_CAMERA_URL = "http://192.168.185.245:8080/video"  # 需包含视频流路径


# --------------------- 工具函数 ---------------------
def create_face_data_dir():
    if not os.path.exists(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR)


def load_user_db():
    """加载用户数据库"""
    global USER_DB, FACE_ID_COUNTER
    if os.path.exists(USER_DB_PATH):
        try:
            with open(USER_DB_PATH, "rb") as f:
                USER_DB = pickle.load(f)
            # 初始化FACE_ID_COUNTER为最大face_id+1
            if USER_DB:
                max_id = max(user[1] for user in USER_DB.values())
                FACE_ID_COUNTER = max_id + 1
        except Exception as e:
            print(f"加载用户数据库失败: {e}")
            USER_DB = {}
    else:
        USER_DB = {}


def save_user_db():
    """保存用户数据库"""
    try:
        with open(USER_DB_PATH, "wb") as f:
            pickle.dump(USER_DB, f)
        print("用户数据库保存成功")
    except Exception as e:
        print(f"保存用户数据库失败: {e}")


def load_labels():
    """加载人脸标签"""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_labels(labels):
    """保存人脸标签"""
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)


def train_face_model():
    global FACE_ID_COUNTER, recognizer
    faces = []
    face_ids = []
    labels = load_labels()  # 加载现有标签

    for name in os.listdir(FACE_DATA_DIR):
        person_dir = os.path.join(FACE_DATA_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        if name in labels:
            face_id = labels[name]
        else:
            face_id = FACE_ID_COUNTER
            labels[name] = face_id
            FACE_ID_COUNTER += 1
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                faces.append(gray)
                face_ids.append(face_id)

    if faces:
        recognizer.train(faces, np.array(face_ids))
        recognizer.save(MODEL_PATH)  # 使用save而不是write
        save_labels(labels)  # 保存更新后的标签
        print("人脸模型训练并保存成功")
    else:
        print("无新人脸数据，跳过训练")


def capture_face(username, camera_type):
    """
    camera_type: "pc"（电脑摄像头） / "phone"（手机IP摄像头）
    """
    create_face_data_dir()
    person_dir = os.path.join(FACE_DATA_DIR, username)
    os.makedirs(person_dir, exist_ok=True)

    # 选择摄像头
    if camera_type == "pc":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(PHONE_CAMERA_URL)  # 使用手机摄像头地址

    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return False

    count = 0
    print(f"开始为 {username} 采集人脸数据，按 'q' 结束...")
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (100, 100))
            img_path = os.path.join(person_dir, f"{count}.png")
            cv2.imwrite(img_path, face_roi)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("人脸采集", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return count >= 50


def recognize_face(username=None):
    """
    如果提供username，则验证指定用户；否则尝试识别所有已注册用户
    返回: (成功/失败, 识别出的用户名)
    """
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("错误", "人脸模型未训练！请先注册")
        return False, None

    # 加载人脸标签
    labels = load_labels()
    if not labels:
        messagebox.showerror("错误", "人脸标签未找到！请先注册")
        return False, None

    id_to_name = {v: k for k, v in labels.items()}

    # 选择摄像头类型
    camera_choice = simpledialog.askstring(
        "摄像头选择", "请输入验证摄像头类型（pc/phone）:",
        initialvalue="pc"
    )
    if camera_choice not in ["pc", "phone"]:
        messagebox.showerror("错误", "无效选择！请输入 pc 或 phone")
        return False, None

    # 选择摄像头
    if camera_choice == "pc":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(PHONE_CAMERA_URL)  # 使用手机摄像头地址

    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return False, None

    # 加载人脸识别模型
    recognizer.read(MODEL_PATH)

    success = False
    recognized_username = None
    print(f"开始人脸验证，按 'q' 结束...")

    start_time = time.time()
    while time.time() - start_time < 10:  # 设置10秒超时
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_id, confidence = recognizer.predict(face_roi)

            if confidence > 80:  # 置信度阈值，值越小越严格
                continue

            recognized_username = id_to_name.get(face_id, None)

            # 如果指定了username，验证是否匹配
            if username and recognized_username == username:
                success = True
                break
            # 如果没有指定username，只要识别出已注册用户就算成功
            elif not username and recognized_username in USER_DB:
                success = True
                break

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if success:
            break

        cv2.imshow("人脸验证", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if success and recognized_username:
        return True, recognized_username
    else:
        return False, None


def realtime_recognition(camera_type):
    """
    camera_type: "pc" / "phone"
    """
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("错误", "人脸模型未训练！无法监控")
        return

    # 加载人脸标签
    labels = load_labels()
    if not labels:
        messagebox.showerror("错误", "人脸标签未找到！无法监控")
        return

    id_to_name = {v: k for k, v in labels.items()}

    # 选择摄像头
    if camera_type == "pc":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(PHONE_CAMERA_URL)  # 使用手机摄像头地址

    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return

    # 加载人脸识别模型
    recognizer.read(MODEL_PATH)

    print("开始实时人脸识别监控，按 'q' 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_id, confidence = recognizer.predict(face_roi)
            name = id_to_name.get(face_id, "未知")
            if confidence < 80:
                text = f"{name} (可信度: {int(100 - confidence)}%)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                text = "未知人员"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("实时人脸识别监控", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# --------------------- 界面逻辑 ---------------------
def register_user():
    username = simpledialog.askstring("注册", "请输入用户名:")
    if not username:
        return
    if username in USER_DB:
        messagebox.showerror("错误", "用户名已存在！")
        return

    password = simpledialog.askstring("注册", "请输入密码:", show="*")
    if not password:
        return

    # 让用户选择采集摄像头类型
    camera_choice = simpledialog.askstring(
        "摄像头选择", "请输入采集摄像头类型（pc/phone）:",
        initialvalue="pc"
    )
    if camera_choice not in ["pc", "phone"]:
        messagebox.showerror("错误", "无效选择！请输入 pc 或 phone")
        return

    if not capture_face(username, camera_choice):
        messagebox.showerror("错误", "人脸采集失败，注册终止！")
        return

    train_face_model()

    # 获取新用户的face_id
    labels = load_labels()
    face_id = labels.get(username, None)

    if face_id is not None:
        USER_DB[username] = (password, face_id, None)  # 初始登录方式为None
        save_user_db()  # 保存用户数据库
        messagebox.showinfo("成功", "注册成功！")
    else:
        messagebox.showerror("错误", "无法获取人脸ID，注册失败！")


def login_user():
    # 让用户选择登录方式
    login_method = simpledialog.askstring(
        "登录方式", "请选择登录方式（face/account）:",
        initialvalue="face"
    )

    if login_method not in ["face", "account"]:
        messagebox.showerror("错误", "无效选择！请输入 face 或 account")
        return

    if login_method == "face":
        success, username = recognize_face()
        if success and username:
            # 检查人脸对应的用户是否存在
            if username in USER_DB:
                messagebox.showinfo("成功", f"欢迎您，{username}！（人脸识别登录）")
                # 更新用户数据库中的最后登录方式
                USER_DB[username] = (USER_DB[username][0], USER_DB[username][1], "face")
                save_user_db()

                # 让用户选择监控摄像头类型
                monitor_camera_choice = simpledialog.askstring(
                    "摄像头选择", "请输入监控摄像头类型（pc/phone）:",
                    initialvalue="pc"
                )
                if monitor_camera_choice not in ["pc", "phone"]:
                    messagebox.showerror("错误", "无效选择！请输入 pc 或 phone")
                    return

                root.destroy()
                realtime_recognition(monitor_camera_choice)
            else:
                messagebox.showerror("错误", "该用户未注册！")
        else:
            messagebox.showerror("错误", "人脸验证失败！")
            # 人脸验证失败后，给用户机会使用账号密码登录
            if messagebox.askyesno("提示", "是否尝试使用账号密码登录？"):
                login_user()  # 递归调用，让用户重新选择登录方式
    else:  # account
        username = entry_username.get()
        password = entry_password.get()

        if not username or not password:
            messagebox.showerror("错误", "用户名和密码不能为空！")
            return

        if username not in USER_DB:
            messagebox.showerror("错误", "用户名不存在！")
            return
        if USER_DB[username][0] != password:
            messagebox.showerror("错误", "密码错误！")
            return

        messagebox.showinfo("成功", f"欢迎您，{username}！（账号密码登录）")
        # 更新用户数据库中的最后登录方式
        USER_DB[username] = (USER_DB[username][0], USER_DB[username][1], "account")
        save_user_db()

        # 让用户选择监控摄像头类型
        monitor_camera_choice = simpledialog.askstring(
            "摄像头选择", "请输入监控摄像头类型（pc/phone）:",
            initialvalue="pc"
        )
        if monitor_camera_choice not in ["pc", "phone"]:
            messagebox.showerror("错误", "无效选择！请输入 pc 或 phone")
            return

        root.destroy()
        realtime_recognition(monitor_camera_choice)


def exit_app():
    root.destroy()


# --------------------- 创建主界面 ---------------------
# 程序启动时加载用户数据库
load_user_db()

# 如果模型文件存在，加载模型
if os.path.exists(MODEL_PATH):
    recognizer.read(MODEL_PATH)

root = tk.Tk()
root.title("人脸识别系统")
root.geometry("400x300")

label_title = tk.Label(root, text="人脸识别系统", font=("黑体", 16, "bold"))
label_title.pack(pady=20)

label_username = tk.Label(root, text="用户名:")
label_username.pack(pady=5)
entry_username = tk.Entry(root)
entry_username.pack(pady=5)

label_password = tk.Label(root, text="密码:")
label_password.pack(pady=5)
entry_password = tk.Entry(root, show="*")
entry_password.pack(pady=5)

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=20)

button_login = tk.Button(frame_buttons, text="登录", command=login_user)
button_login.pack(side="left", padx=10)

button_register = tk.Button(frame_buttons, text="注册", command=register_user)
button_register.pack(side="left", padx=10)

button_exit = tk.Button(frame_buttons, text="退出", command=exit_app)
button_exit.pack(side="left", padx=10)

root.mainloop()