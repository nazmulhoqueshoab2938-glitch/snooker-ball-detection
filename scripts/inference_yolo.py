import os
from ultralytics import YOLO

model_path = 'runs/detect/snooker_v1/weights/best.pt'

if os.path.exists(model_path):
    model = YOLO(model_path)
    # আপনার টেস্ট ইমেজের পাথ দিন
    source_image = 'test.jpg' 
    
    if os.path.exists(source_image):
        results = model.predict(source=source_image, conf=0.25, save=True)
        print("ডিটেকশন সম্পন্ন হয়েছে! রেজাল্ট runs/detect পেজে সেভ হয়েছে।")
    else:
        print("টেস্ট ইমেজটি পাওয়া যায়নি।")
else:
    print("প্রশিক্ষিত মডেলের ফাইল পাওয়া যায়নি। আগে ট্রেইনিং শেষ করুন।")
