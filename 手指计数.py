import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


class HandGestureRecognition:
    def __init__(self):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 手指关节点ID
        self.finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖
        self.finger_pips = [3, 6, 10, 14, 18]  # 手指的第二关节点

        # 尝试加载中文字体
        try:
            # Windows系统字体路径
            self.font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 24)
            self.small_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 18)
        except:
            try:
                # 备选字体路径
                self.font = ImageFont.truetype("arial.ttf", 24)
                self.small_font = ImageFont.truetype("arial.ttf", 18)
            except:
                # 使用默认字体
                self.font = ImageFont.load_default()
                self.small_font = ImageFont.load_default()

        # 手势历史记录用于平滑
        self.gesture_history = []
        self.max_history = 5

    def put_chinese_text(self, img, text, position, font, color=(255, 255, 255)):
        """在图像上绘制中文文字（优化版）"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)

    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

    def is_thumb_up(self, landmarks):
        """改进的拇指检测方法"""
        # 获取关键点
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]  # 拇指的第二关节
        thumb_mcp = landmarks[2]  # 拇指的掌指关节
        index_mcp = landmarks[5]  # 食指的掌指关节

        # 计算距离
        distance_tip_ip = self.calculate_distance(thumb_tip, thumb_ip)
        distance_tip_mcp = self.calculate_distance(thumb_tip, thumb_mcp)
        distance_tip_index = self.calculate_distance(thumb_tip, index_mcp)

        # 改进的检测逻辑
        if distance_tip_ip > distance_tip_mcp * 0.7:
            return True
        if distance_tip_index > distance_tip_mcp * 0.9:
            return True
        return False

    def count_fingers(self, landmarks, handedness):
        """计算伸出的手指数量（优化版）"""
        fingers = []

        # 改进的拇指检测
        fingers.append(1 if self.is_thumb_up(landmarks) else 0)

        # 其他四个手指：使用距离检测代替坐标比较
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            pip = landmarks[self.finger_pips[i]]
            dip = landmarks[self.finger_pips[i] - 1]  # 手指的第一关节

            # 计算指尖到第一关节的距离
            distance_tip_dip = self.calculate_distance(tip, dip)
            distance_dip_pip = self.calculate_distance(dip, pip)

            # 如果指尖到第一关节的距离大于第一关节到第二关节的距离，则手指是伸直的
            if distance_tip_dip > distance_dip_pip * 0.8:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers), fingers

    def draw_finger_info(self, frame, finger_status):
        """绘制手指状态信息"""
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        for i, (name, status) in enumerate(zip(finger_names, finger_status)):
            text = f'{name}: {"伸出" if status else "弯曲"}'
            color = (0, 255, 0) if status else (255, 0, 0)  # RGB格式
            frame = self.put_chinese_text(frame, text, (10, 30 + i * 35),
                                          self.small_font, color)
        return frame

    def run(self):
        """运行手势识别（优化版）"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 手势映射
        gestures = {
            0: "拳头",
            1: "一",
            2: "二/胜利",
            3: "三/OK",
            4: "四",
            5: "五/张开手掌"
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 水平翻转图像，使手势更自然
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # 转换BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 处理图像
            results = self.hands.process(rgb_frame)

            finger_count = 0
            finger_status = [0, 0, 0, 0, 0]
            handedness = "Unknown"

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 获取手的方向信息
                    if results.multi_handedness:
                        handedness = results.multi_handedness[hand_idx].classification[0].label

                    # 绘制手部关键点
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 计算手指数量
                    finger_count, finger_status = self.count_fingers(hand_landmarks.landmark, handedness)

                    # 绘制手指状态信息
                    frame = self.draw_finger_info(frame, finger_status)

                    # 显示手的方向
                    direction_text = f'手型: {handedness}'
                    frame = self.put_chinese_text(frame, direction_text, (w - 150, 60),
                                                  self.small_font, (200, 50, 200))

            # 手势平滑处理（防止频繁跳动）
            self.gesture_history.append(finger_count)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)
            smoothed_count = max(set(self.gesture_history),
                                 key=self.gesture_history.count) if self.gesture_history else 0

            # 显示手指总数
            frame = self.put_chinese_text(frame, f'手指数量: {smoothed_count}',
                                          (10, 200), self.font, (0, 0, 255))

            # 根据手指数量显示不同信息
            gesture = gestures.get(smoothed_count, "未知")
            frame = self.put_chinese_text(frame, f'手势: {gesture}',
                                          (10, 250), self.font, (255, 255, 0))

            # 显示使用说明
            frame = self.put_chinese_text(frame, '按 ESC 退出',
                                          (w - 120, 30), self.small_font, (255, 255, 255))

            # 添加边框增强视觉效果
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 255), 2)

            # 在右上角显示FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_text = f'FPS: {fps:.1f}' if fps > 0 else 'FPS: N/A'
            frame = self.put_chinese_text(frame, fps_text, (w - 150, 30),
                                          self.small_font, (200, 200, 0))

            cv2.imshow('手势识别系统', frame)

            # 按ESC键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# 优化后的英文版本
class HandGestureRecognitionEnglish:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]

        # 手势历史记录用于平滑
        self.gesture_history = []
        self.max_history = 5

    def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

    def is_thumb_up(self, landmarks):
        """改进的拇指检测方法"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]

        distance_tip_ip = self.calculate_distance(thumb_tip, thumb_ip)
        distance_tip_mcp = self.calculate_distance(thumb_tip, thumb_mcp)
        distance_tip_index = self.calculate_distance(thumb_tip, index_mcp)

        if distance_tip_ip > distance_tip_mcp * 0.7:
            return True
        if distance_tip_index > distance_tip_mcp * 0.9:
            return True
        return False

    def count_fingers(self, landmarks, handedness):
        """计算伸出的手指数量（优化版）"""
        fingers = []

        # 拇指检测
        fingers.append(1 if self.is_thumb_up(landmarks) else 0)

        # 其他四个手指
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            pip = landmarks[self.finger_pips[i]]
            dip = landmarks[self.finger_pips[i] - 1]

            distance_tip_dip = self.calculate_distance(tip, dip)
            distance_dip_pip = self.calculate_distance(dip, pip)

            if distance_tip_dip > distance_dip_pip * 0.8:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers), fingers

    def run(self):
        """运行手势识别（优化版）"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 手势映射
        gestures = {
            0: "Fist",
            1: "One",
            2: "Two/Peace",
            3: "Three/OK",
            4: "Four",
            5: "Five/Open Hand"
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            finger_count = 0
            finger_status = [0, 0, 0, 0, 0]
            handedness = "Unknown"

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 获取手的方向信息
                    if results.multi_handedness:
                        handedness = results.multi_handedness[hand_idx].classification[0].label

                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    finger_count, finger_status = self.count_fingers(hand_landmarks.landmark, handedness)

                    # Display finger status
                    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                    for i, (name, status) in enumerate(zip(finger_names, finger_status)):
                        color = (0, 255, 0) if status else (0, 0, 255)
                        cv2.putText(frame, f'{name}: {"Up" if status else "Down"}',
                                    (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Display handedness
                    cv2.putText(frame, f'Hand: {handedness}', (w - 150, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 200), 1)

            # 手势平滑处理
            self.gesture_history.append(finger_count)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)
            smoothed_count = max(set(self.gesture_history),
                                 key=self.gesture_history.count) if self.gesture_history else 0

            # Display finger count
            cv2.putText(frame, f'Finger Count: {smoothed_count}', (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display gesture
            gesture = gestures.get(smoothed_count, "Unknown")
            cv2.putText(frame, f'Gesture: {gesture}', (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display instructions
            cv2.putText(frame, 'Press ESC to exit', (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Add border
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 255), 2)

            # Display FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_text = f'FPS: {fps:.1f}' if fps > 0 else 'FPS: N/A'
            cv2.putText(frame, fps_text, (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("选择版本:")
    print("1. 中文版本 (需要PIL库)")
    print("2. 英文版本 (纯OpenCV)")

    choice = input("请输入选择 (1 或 2): ")

    if choice == "1":
        try:
            # 检查PIL是否可用
            from PIL import Image, ImageDraw, ImageFont

            recognizer = HandGestureRecognition()
            recognizer.run()
        except ImportError:
            print("PIL库未安装，请运行: pip install Pillow")
            print("使用英文版本...")
            recognizer = HandGestureRecognitionEnglish()
            recognizer.run()
    else:
        recognizer = HandGestureRecognitionEnglish()
        recognizer.run()