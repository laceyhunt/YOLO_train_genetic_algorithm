from ultralytics import YOLO
import os
import contextlib
import sys
# os.environ['YOLO_VERBOSE'] = 'False'
# model = YOLO("yolov8n.pt") # load a pretrained model (for transfer learning)
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data="new_train_set/data.yaml", patience=50,epochs=500)
# results = model.train(data="dataset_reorganized/data.yaml", epochs=100, imgsz=640)


log_file = open("train_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file


model = YOLO("yolov8n.pt") # load a pretrained model (for transfer learning)
results = model.train(data='indivs/indiv_0/data.yaml', 
                    patience=50, 
                    epochs=50,
                    verbose=False,
                    save=False,
                    plots=False,
                    exist_ok=True)
# os.path.join(test_dir, 'test_data.yaml')
# restore after
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

sys.stdout = log_file
sys.stderr = log_file

metrics = model.val(
        # data='shared_test_set/test.yaml',
        data=f'test_dir/test_data.yaml',
        split='test',
        plots=False,
        verbose=False
    )
fitness = metrics.maps.mean() # maps is for each image category so i take mean to account for all


# restore after
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()

print(" ***** METRICS ****")
print(model.names) # for class labels for the following mAPs...
print(metrics.maps)
print("type of maps=")
print(type(metrics.maps))
class_names = model.names  # dict: {0: "Blackbean", 1: "Flax", ...}
class_order = [class_names[i] for i in range(len(class_names))]
print(f"names: {class_names} \n in order names: {class_order}")
print(f"Average= {fitness}")

