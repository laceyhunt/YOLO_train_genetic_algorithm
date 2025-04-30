import os 
import shutil
import random
from collections import defaultdict


# Paths to original dataset
root_dir = 'dataset'
new_root = 'dataset_reorganized'

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

# Split ratios
train_ratio = 0.8
val_ratio = 0.2 # 20% of training
test_ratio = 0.1 # 10% of total, 112 imgs

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

# print(category_path)
# print(category_class_names)
# print(class_names)
# print(class_files)


# balance the num of each class type for train, test, val
for category, files in class_files.items():
    random.shuffle(files)
    # Determine the number of files per set (train, val, test)
    total_files = len(files)
    test_size = int(total_files * test_ratio)  # 10% for the test set
    train_size = int(total_files-test_size)  # 90% for the training set
    train_files = files[:train_size]  # First 90% of the files for training
    test_files=files[train_size:]     # Rest for test set

    # validation is 20% of the training set
    val_size = int(train_size * val_ratio)  # 20% of training set for validation
    val_files = train_files[:val_size]  # First 20% for validation
    train_files = train_files[val_size:]  # Remaining 80% for training

    print(f'\nFor category {category}: train = {train_size}, test = {test_size}, validation = {val_size}')
    print(f'Checking category {category}: train = {len(train_files)}, validation = {len(val_files)}, test = {len(test_files)}')

    # Check if test_files has content
    if not test_files:
        print("Warning: Test set is empty. Something went wrong with the split.")
        break

    # Move files to the appropriate directories
    for image_path, label_path in train_files:
        _=shutil.copy(image_path, f'{new_root}/images/train/{os.path.basename(image_path)}')
        _=shutil.copy(label_path, f'{new_root}/labels/train/{os.path.basename(label_path)}')

    for image_path, label_path in val_files:
        _=shutil.copy(image_path, f'{new_root}/images/val/{os.path.basename(image_path)}')
        _=shutil.copy(label_path, f'{new_root}/labels/val/{os.path.basename(label_path)}')

    for image_path, label_path in test_files:
        _=shutil.copy(image_path, f'{new_root}/images/test/{os.path.basename(image_path)}')
        _=shutil.copy(label_path, f'{new_root}/labels/test/{os.path.basename(label_path)}')

    print(f"Test directory now contains {len(os.listdir(f'{new_root}/images/test'))} files.")
    print(f"Train directory now contains {len(os.listdir(f'{new_root}/images/train'))} files.")
    print(f"Validation directory now contains {len(os.listdir(f'{new_root}/images/val'))} files.")

# Write data.yaml file
class_names=list(class_names)
with open(f'{new_root}/data.yaml', 'w') as f:
    f.write(f"path: ./dataset_reorganized\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    # f.write(f"nc: {len(class_names)}\n")
    f.write(f"names:\n")# {list(class_names)}\n")
    for i, name in enumerate(class_names):
        f.write(f"   {i}: {class_names[i]}\n")

print("Reorganization completed with balanced splits!")


print(new_root)