import time

import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

""" 配置 """
window_title = "Object Detector"
is_flip = True

# 相机
width = 1920
height = 1080

""" 标签 """
LABEL_NO_CAMERA = "Please check your video capturing device."
LABEL_FPS = "FPS: %.1f"
LABEL_ACTION = f"Action %d: %.2fs"

""" 文本样式 """
MARGIN = 10  # pixels
ROW_SIZE = 16  # pixels
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
    base_options = python.BaseOptions(model_asset_path="exported_model/object/object.tflite")
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        max_results=2,
        score_threshold=0.35,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.ObjectDetector.create_from_options(options)

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

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # 识别
        results = detector.detect_for_video(mp_image, int(round(frame_timestamp_ms)))
        # 画框
        annotated_image = visualize(image, results)

        # 帧率
        time_consuming = (time.time() - cost_start)
        time_consuming_per = round(1 / time_consuming, 0)
        time_consuming_per = time_consuming_per if (time_consuming_per <= fps) else fps
        cv2.putText(image, LABEL_FPS % time_consuming_per, XY_FPS, FONT_FACE, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        # 还原颜色
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
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
    cap.release()


def visualize(
        image,
        detection_result
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, FONT_FACE,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


if __name__ == "__main__":
    detect()
