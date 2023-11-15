import os
import json
import tensorflow as tf

assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

train_dataset_path = "eye_target/train"
validation_dataset_path = "eye_target/validation"

# 查看标签
with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
    labels_json = json.load(f)
for category_item in labels_json["categories"]:
    print(f"{category_item['id']}: {category_item['name']}")

# 创建数据集
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

# 设置参数
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(export_dir='exported_model/object', epochs=50, batch_size=5)
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams,
)

# 训练模型
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)

# 评估模型性能
loss, coco_metrics = model.evaluate(validation_data, batch_size=3)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

# 导出 Tensorflow Lite 模型
model.export_model('object.tflite')
exit(0)
#
# # 量化感知训练（int8 量化）
# qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
# model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
# qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# print(f"QAT validation loss: {qat_loss}")
# print(f"QAT validation coco metrics: {qat_coco_metrics}")
#
# new_qat_hparams = object_detector.QATHParams(learning_rate=0.9, batch_size=4, epochs=15, decay_steps=5, decay_rate=0.96)
# model.restore_float_ckpt()
# model.quantization_aware_training(train_data, validation_data, qat_hparams=new_qat_hparams)
# qat_loss, qat_coco_metrics = model.evaluate(validation_data)
# print(f"QAT validation loss: {qat_loss}")
# print(f"QAT validation coco metrics: {qat_coco_metrics}")
#
# # 导出 Tensorflow Lite int8 量化模型
# model.export_model('object_int8_qat.tflite')
