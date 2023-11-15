import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 16  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


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
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def detect():
    # 配置模型
    base_options = python.BaseOptions(model_asset_path="exported_model/object/object.tflite")
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        max_results=2,
        score_threshold=0.35,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    # 当前第几帧
    frame_index = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("failed!!!")
            continue
        frame_index += 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_timestamp_ms = 1000 * frame_index / fps

        # 镜像翻转
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # 识别
        results = detector.detect_for_video(mp_image, int(round(frame_timestamp_ms)))
        # 画框
        annotated_image = visualize(image, results)
        # 还原颜色
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        # 展示
        cv2.imshow("Object Detector", rgb_annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


if __name__ == "__main__":
    detect()
