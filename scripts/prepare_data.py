import os
import json
import shutil

# আপনার সুপারভাইজলি ডেটা ফোল্ডারের পথ
SVE_DATASET_ROOT = 'data/supervisely_export' 
OUTPUT_DIR = 'yolo_data' 

CLASSES = ['Black', 'Blue', 'Brown', 'Green', 'Pink', 'Red', 'White', 'Yellow'] 
CLASS_MAP = {cls: i for i, cls in enumerate(CLASSES)}

for subset in ['images', 'labels']:
    os.makedirs(os.path.join(OUTPUT_DIR, subset, 'train'), exist_ok=True)

print("ডেটা রূপান্তর শুরু হচ্ছে...")

image_count = 0
for annotation_root, _, files in os.walk(os.path.join(SVE_DATASET_ROOT, 'ann')):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(annotation_root, file)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = data['image_name']
            
            possible_image_paths = [
                os.path.join(SVE_DATASET_ROOT, 'img', image_name),
                os.path.join(SVE_DATASET_ROOT, 'original_img', image_name),
            ]
            
            image_src_path = next((p for p in possible_image_paths if os.path.exists(p)), None)
            if not image_src_path:
                continue

            shutil.copy2(image_src_path, os.path.join(OUTPUT_DIR, 'images', 'train', image_name))
            image_count += 1
            
            yolo_labels = []
            for obj in data['objects']:
                class_name = obj['classTitle']
                if class_name in CLASS_MAP and obj['geometryType'] == 'rectangle':
                    class_id = CLASS_MAP[class_name]
                    points = obj['points']['exterior']
                    x_min, y_min = points[0]
                    x_max, y_max = points[1]
                    
                    img_width = data['size']['width']
                    img_height = data['size']['height']
                    
                    x_center = max(0, min(1, (x_min + x_max) / 2 / img_width))
                    y_center = max(0, min(1, (y_min + y_max) / 2 / img_height))
                    width = max(0, min(1, (x_max - x_min) / img_width))
                    height = max(0, min(1, (y_max - y_min) / img_height))

                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            label_file_name = os.path.splitext(image_name)[0] + '.txt'
            with open(os.path.join(OUTPUT_DIR, 'labels', 'train', label_file_name), 'w') as f:
                f.write('\n'.join(yolo_labels))

print(f"মোট {image_count}টি ছবি রূপান্তর সম্পন্ন হয়েছে।")
