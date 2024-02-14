from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model(source="data/dev_images/01268.png", show=True, conf=0.5, save=True,project='obj_detect', name='without_captions')

# print("\n\n\n")
# print(results[0])

# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# results = model.train(data='', epochs=100, imgsz=640)








