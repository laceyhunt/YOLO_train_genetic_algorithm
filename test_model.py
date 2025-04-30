from ultralytics import YOLO
import os

# === CONFIGURATION ===
weights_path = 'output/runs/detect/train6/weights/best.pt' 
save_dir = 'output/test6_biggertest_eval'  # Output directory for evaluation results

# Define your test data config inline
test_data = "./dataset_reorganized/data.yaml"#{
#     'test': './new_train_set/images/test',
#     'nc': 13,
#     'names': [
#         'Flax',
#         'Ragweed',
#         'Corn',
#         'Redroot Pigweed',
#         'Blackbean',
#         'Waterhemp',
#         'Horseweed',
#         'Kochia',
#         'Lentil',
#         'Sugar beet',
#         'Field Pea',
#         'Canola',
#         'Soybean'
#     ]
# }

# === EVALUATE MODEL ===
print(f"Loading model from: {weights_path}")
model = YOLO(weights_path)

print(f"Evaluating model on ./dataset_reorganized/images/test ...")
metrics = model.val(
    data=test_data,
    split='test',
    save=True,
    project=save_dir,
    name='eval',
    exist_ok=True
)

# === SAVE METRICS TO FILE ===
metrics_path = os.path.join(save_dir, 'eval', 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(str(metrics))
print(metrics.box.maps)  # list of mAP50-95 for each category
print(f"Done. Metrics saved to {metrics_path}")
