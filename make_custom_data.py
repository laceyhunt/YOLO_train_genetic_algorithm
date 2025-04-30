import os 
import shutil
import random
from collections import defaultdict


# Paths to original dataset
root_dir = 'dataset'
new_root = 'new_train_set'

print(root_dir)

# Create directories for new structure
os.makedirs(f'{new_root}/images/train', exist_ok=True)
os.makedirs(f'{new_root}/images/val', exist_ok=True)
os.makedirs(f'{new_root}/images/test', exist_ok=True)
os.makedirs(f'{new_root}/labels/train', exist_ok=True)
os.makedirs(f'{new_root}/labels/val', exist_ok=True)
os.makedirs(f'{new_root}/labels/test', exist_ok=True)

# To collect all class names and their corresponding files
class_names = set()
class_files = defaultdict(list)

print(class_names)
print(class_files)

# Split sizes
test_size=20
indiv_size=100
train_ratio=0.8
val_ratio=0.2

print(os.listdir(root_dir))

for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    # if path exists
    if os.path.isdir(category_path):
        # read class labels
        with open(os.path.join(category_path,'classes.txt'),'r') as f:
            category_class_names = f.read().splitlines()
            class_names.update(category_class_names)
        # process image and text label in dir
        for filename in os.listdir(category_path):
            if filename.endswith('.JPG'):
                img_path = os.path.join(category_path, filename)
                label_path = img_path.replace('.JPG','.txt')
                class_files[category].append((img_path, label_path))

# Gather all image-label pairs from all categories
all_files = []

for category, files in class_files.items():
    all_files.extend(files)

random.shuffle(all_files)

# Global split
total_needed = indiv_size + test_size
if len(all_files) < total_needed:
    raise ValueError(f"Not enough total data to meet requested sizes. Have {len(all_files)}, need {total_needed}")

train_val_files = all_files[:indiv_size]
test_files = all_files[indiv_size:indiv_size + test_size]

val_size = int(indiv_size * val_ratio)
val_files = train_val_files[:val_size]
train_files = train_val_files[val_size:]

print(f"Final Split Sizes:\nTrain: {len(train_files)}\nVal: {len(val_files)}\nTest: {len(test_files)}")

# Move files
def move_files(file_list, subset):
    for image_path, label_path in file_list:
        if not os.path.exists(label_path): continue
        shutil.copy(image_path, f'{new_root}/images/{subset}/{os.path.basename(image_path)}')
        shutil.copy(label_path, f'{new_root}/labels/{subset}/{os.path.basename(label_path)}')

move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

# Write data.yaml file
class_names=list(class_names)
with open(f'{new_root}/data.yaml', 'w') as f:
    f.write(f"path: ./{new_root}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    # f.write(f"nc: {len(class_names)}\n")
    f.write(f"names:\n")# {list(class_names)}\n")
    for i, name in enumerate(class_names):
        f.write(f"   {i}: {class_names[i]}\n")

print("Reorganization completed with custom splits!")


print(new_root)