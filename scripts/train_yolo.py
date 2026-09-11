from ultralytics import YOLO

CLASSES = ['Black', 'Blue', 'Brown', 'Green', 'Pink', 'Red', 'White', 'Yellow']

data_yaml_content = f"""
path: yolo_data
train: images/train
val: images/train
nc: {len(CLASSES)}
names: {CLASSES}
"""

with open("snooker_data.yaml", "w") as f:
    f.write(data_yaml_content)

# YOLOv8s মডেল ব্যবহার করা হচ্ছে
model = YOLO('yolov8s.pt')

print("ট্রেনিং শুরু হচ্ছে...")
model.train(
    data='snooker_data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='snooker_v1',
    device=0
)
print("ট্রেনিং সম্পন্ন হয়েছে!")
