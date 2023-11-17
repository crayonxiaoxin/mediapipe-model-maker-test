import time
from os import path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

""" 配置 """
window_title = "Gesture Detector"
hands_count = 2
is_flip = True

# 相机
width = 1920
height = 1080

""" MediaPipe """
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

""" 模型 """
MODEL_HAND_GESTURE = path.abspath(path.join(path.dirname(__file__), "exported_model/gesture/gesture_recognizer.task"))

""" 标签 """
LABEL_NO_CAMERA = "Please check your video capturing device."
LABEL_FPS = "FPS: %.1f"
LABEL_ACTION = f"Action %d: %.2fs"

""" 文本样式 """
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
ACTION_TEXT_COLOR = (255, 0, 255)

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 0, 0)  # red
TEXT_COLOR_PROMPT = (0, 0, 255)
TEXT_COLOR_GREEN = (0, 255, 0)
XY_FPS = (50, 50)


def detect(error_callback=None, screen_size: tuple | None = None):
    global is_flip

    # 配置模型
    base_options = python.BaseOptions(model_asset_path=MODEL_HAND_GESTURE, delegate="GPU")
    hand_options = vision.GestureRecognizerOptions(base_options=base_options,
                                                   num_hands=hands_count,
                                                   running_mode=vision.RunningMode.VIDEO, )
    recognizer = vision.GestureRecognizer.create_from_options(hand_options)

    # 窗口位置
    cv2.namedWindow(window_title)
    if screen_size is not None:
        x = (screen_size[0] - width) / 2
        y = (screen_size[1] - height) / 2
        cv2.moveWindow(window_title, abs(int(x)), abs(int(y)))

    # 开启摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 如果没有摄像头
    if fps == 0:
        if error_callback is not None:
            error_callback(LABEL_NO_CAMERA)
        else:
            print(LABEL_NO_CAMERA)
        cap.release()
        return

    # 当前第几帧
    frame_index = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("failed!!!")
            continue
        cost_start = time.time()
        frame_index += 1
        frame_timestamp_ms = int(round(1000 * frame_index / fps))

        # 颜色转换 & 镜面翻转
        if is_flip:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为 MediaPipe 可识别格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # 识别
        results_gesture = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

        # 画图
        image = draw_landmarks_on_image(image, results_gesture)

        # 右侧标题
        _title(image, window_title)

        # 帧率
        time_consuming = (time.time() - cost_start)
        time_consuming_per = round(1 / time_consuming, 0)
        time_consuming_per = time_consuming_per if (time_consuming_per <= fps) else fps
        cv2.putText(image, LABEL_FPS % time_consuming_per, XY_FPS, FONT_FACE, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        # 还原颜色
        rgb_annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 展示
        cv2.imshow(window_title, rgb_annotated_image)

        # 监听按键
        wait_key = cv2.waitKey(5) & 0xFF
        if wait_key == 27 or wait_key == ord('q'):  # ESC/Q - 退出
            break
        elif wait_key == ord('f'):  # F - 镜像翻转
            is_flip = not is_flip
        elif wait_key == ord('r'):  # R - 重置
            print("r")

    cv2.destroyAllWindows()
    # 释放相机
    cap.release()
    return


def draw_landmarks_on_image(rgb_image, detection_result):
    gestures_list = detection_result.gestures
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    if len(annotated_image.shape) < 2:
        return annotated_image

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        hand = handedness[0].category_name

        # 如果翻转
        if is_flip:
            if hand == "Left":
                hand = "Right"
            elif hand == "Right":
                hand = "Left"

        print(hand, gesture[0].category_name)
        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{hand} {gesture[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def _title(image, label):
    text_width, _ = measure_text(label, FONT_FACE, FONT_SIZE, FONT_THICKNESS)
    cv2.putText(image, label, (width - text_width - 50, 50), FONT_FACE, FONT_SIZE, TEXT_COLOR_PROMPT,
                FONT_THICKNESS)


def measure_text(label, font_face, font_scale, font_thickness):
    text_measure = cv2.getTextSize(label, font_face, font_scale, font_thickness)
    text_width = text_measure[0][0]
    text_height = text_measure[0][1]
    return text_width, text_height


if __name__ == '__main__':
    detect()
