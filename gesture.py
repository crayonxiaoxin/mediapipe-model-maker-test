import os

from mediapipe_model_maker import gesture_recognizer

dataset_path = "rps_data_sample"

# 打印标签
print(dataset_path)
labels = []
for i in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, i)):
        labels.append(i)
print(labels)

# 加载数据集
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# 训练模型
hparams = gesture_recognizer.HParams(export_dir="exported_model/gesture")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# 评估模型性能
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

# 导出 Tensorflow Lite 模型
model.export_model()
