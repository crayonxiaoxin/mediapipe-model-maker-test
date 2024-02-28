import os
import json
import tensorflow as tf

from mediapipe_model_maker import object_detector
from dev import object_detector_for_mac

assert tf.__version__.startswith('2')

# 配置
train_dataset_path = "ruler_target/train"
validation_dataset_path = "ruler_target/validation"
export_model_dir = "exported_model/object"
export_model_name = "object"
train_learning_rate = 0.3
train_epochs = 30
train_batch_size = 8
validation_batch_size = 3


def train(
        learning_rate=train_learning_rate,
        epochs=train_epochs,
        batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        export_dir=export_model_dir,
        export_name=export_model_name,
        train_dataset_dir=train_dataset_path,
        validation_dataset_dir=validation_dataset_path
):
    # 查看标签
    with open(os.path.join(train_dataset_dir, "labels.json"), "r") as f:
        labels_json = json.load(f)
    for category_item in labels_json["categories"]:
        print(f"{category_item['id']}: {category_item['name']}")

    # 创建数据集
    train_data = object_detector.Dataset.from_coco_folder(
        train_dataset_dir,
        cache_dir='/tmp/od_data/train'
    )
    validation_data = object_detector.Dataset.from_coco_folder(
        validation_dataset_dir,
        cache_dir='/tmp/od_data/validation'
    )
    print("train_data size: ", train_data.size)
    print("validation_data size: ", validation_data.size)

    # 设置参数
    spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
    hparams = object_detector.HParams(
        export_dir=export_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams,
    )

    # 训练模型
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options)
    # model = object_detector_for_mac.ObjectDetector.create(
    #     train_data=train_data,
    #     validation_data=validation_data,
    #     options=options
    # )

    # 评估模型性能
    loss, coco_metrics = model.evaluate(
        validation_data,
        batch_size=validation_batch_size
    )
    print("评估模型性能")
    print(f"Validation loss: {loss}")
    print(f"Validation coco metrics: {coco_metrics}")

    # 导出 Tensorflow Lite 模型
    model.export_model(export_name + '.tflite')


if __name__ == "__main__":
    epochs = 60
    learning_rate = 0.26
    version = 3
    train(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=8,
        validation_batch_size=2,
        export_name="ruler_%d_%d_v%d" % (epochs, learning_rate * 100, version)
    )
