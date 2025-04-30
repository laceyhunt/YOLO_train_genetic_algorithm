# MV_trainingset_indivs

import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import os
from ultralytics import YOLO

# Number of images in each training set (Individual)  SET TO A DEBUG VAL FOR NOW
train_sample_size = 20
# Number of images in the shared training set
test_sample_size = 112
# Probability of Mutation (out of 100)
mutation_probability = 5
# Probability of deletion (out of 100)
deletion_probability = 3
# Probability of insertion (out of 100)
insertion_probability = 3
# Number of individuals in a population
pop_size = 10
# Size of the tournament
tournament_size = 5
# Maximum generations for GA
max_generations = 2000
# Number of individuals advanced into the next generation
elitism_num = 5
# Maximum length of an insertion
max_insertion_length = 5
# Maximum length of a deletion
max_deletion_length = 5
# YOLO model training parameters
yolo_max_generations = 500
yolo_patience=50
val_ratio=0.2
pretrained_model='yolov8n.pt'

# Path to original dataset
source_dir = 'dataset_reorganized'
# Path to the new population
new_root = 'indivs'
# Path to new shared test set
test_dir = 'test_dir'
# Remove old directories from last run if they exist before making new ones
if os.path.exists(new_root):
  shutil.rmtree(new_root)
os.makedirs(f'{new_root}')
if os.path.exists(test_dir):
  shutil.rmtree(test_dir)
os.makedirs(f'{test_dir}')
os.makedirs(os.path.join(test_dir, 'images/train'))
os.makedirs(os.path.join(test_dir, 'images/val'))
os.makedirs(os.path.join(test_dir, 'images/test'))
os.makedirs(os.path.join(test_dir, 'labels/train'))
os.makedirs(os.path.join(test_dir, 'labels/val'))
os.makedirs(os.path.join(test_dir, 'labels/test'))

# Start at 0
indiv_num=0

# Get all available train and test image-label pairs from source
train_images_dir = os.path.join(source_dir, 'images/train')
train_labels_dir = os.path.join(source_dir, 'labels/train')
test_images_dir = os.path.join(source_dir, 'images/test')
test_labels_dir = os.path.join(source_dir, 'labels/test')

train_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
print(f"Possible Trains: {len(train_images)}")
print(f"Possible Tests: {len(test_images)}")


# Sample a shared test set
shared_test_images = random.sample(test_images, test_sample_size)

# Save shared test images for calculating fitness
for img_name in shared_test_images:
  label_name = img_name.rsplit('.', 1)[0] + '.txt'
  shutil.copy(os.path.join(test_images_dir, img_name), os.path.join(test_dir, 'images/test', img_name))
  shutil.copy(os.path.join(test_labels_dir, label_name), os.path.join(test_dir, 'labels/test', label_name))

# Write data.yaml for test set
with open(os.path.join(test_dir, 'test_data.yaml'), 'w') as f:
  f.write(f"path: {os.path.abspath(test_dir)}\n")
  f.write("train: images/train\n")
  f.write("val: images/val\n")
  f.write("test: images/test\n")
  f.write("names:\n")

  # Pull class names from original data.yaml
  with open(os.path.join(source_dir, 'data.yaml'), 'r') as original_yaml:
    for line in original_yaml:
      if line.strip().startswith('0:') or line.strip().startswith('1:') or ':' in line.strip()[0:3]:
        f.write("  " + line)


# Individual class, where each individual is a YOLO directory
class Individual:
  # Initialization
  def __init__(self):
    global indiv_num, new_root,train_sample_size,val_ratio,train_images

    self.fitness = 0
    self.number=indiv_num
    self.indiv_dir = os.path.join(new_root, f'indiv_{self.number}')
    self.dir_str = new_root + f"/indiv_{self.number}/data.yaml"

    # Only make train and val since test is shared in the population
    os.makedirs(os.path.join(self.indiv_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(self.indiv_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(self.indiv_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(self.indiv_dir, 'labels/val'), exist_ok=True)

    # Sample and split training set
    sample = random.sample(train_images, train_sample_size)
    val_size = int(train_sample_size * val_ratio)
    val_images = sample[:val_size]
    train_images_subset = sample[val_size:]

    # Copy training set
    for img_name in train_images_subset:
      label_name = img_name.rsplit('.', 1)[0] + '.txt'
      shutil.copy(os.path.join(train_images_dir, img_name), os.path.join(self.indiv_dir, 'images/train', img_name))
      shutil.copy(os.path.join(train_labels_dir, label_name), os.path.join(self.indiv_dir, 'labels/train', label_name))

    # Copy validation set
    for img_name in val_images:
      label_name = img_name.rsplit('.', 1)[0] + '.txt'
      shutil.copy(os.path.join(train_images_dir, img_name), os.path.join(self.indiv_dir, 'images/val', img_name))
      shutil.copy(os.path.join(train_labels_dir, label_name), os.path.join(self.indiv_dir, 'labels/val', label_name))

    # Write data.yaml
    with open(os.path.join(self.indiv_dir, 'data.yaml'), 'w') as f:
      f.write(f"path: {os.path.abspath(self.indiv_dir)}\n")
      f.write("train: images/train\n")
      f.write("val: images/val\n")
      f.write("names:\n")

      # Pull class names from original data.yaml
      with open(os.path.join(source_dir, 'data.yaml'), 'r') as original_yaml:
        for line in original_yaml:
          if line.strip().startswith('0:') or line.strip().startswith('1:') or ':' in line.strip()[0:3]:
            f.write("  " + line)
    # Increment global individual count
    indiv_num+=1
    self.calculate_fitness()

  def calculate_fitness(self):
    # global test_dir
    # model = YOLO("yolov8n.pt") # load a pretrained model (for transfer learning)
    # results = model.train(data=self.dir_str, 
    #                       patience=yolo_patience, 
    #                       epochs=yolo_max_generations,
    #                       verbose=False,
    #                       save=False,
    #                       plots=False,
    #                       exist_ok=True)
    # # os.path.join(test_dir, 'test_data.yaml')
    # metrics = model.val(
    #         # data='shared_test_set/test.yaml',
    #         data=f'{test_dir}/test_data.yaml',
    #         split='test',
    #         plots=False,
    #         verbose=False
    #     )
    # self.fitness = metrics.maps.mean() # maps is for each image category so i take mean to account for all
    # print(" ***** METRICS ****")
    # print(model.names) # for class labels for the following mAPs...
    # print(metrics.maps)
    self.fitness=20
  
  def single_point_crossover(self, p2_dir, child_dir, split):
    p1_dir=self.indiv_dir

    # Helper to get the images from parent
    def get_images(parent_dir, split):
      return [f for f in os.listdir(os.path.join(parent_dir, f'images/{split}')) if f.endswith('.jpg') or f.endswith('.JPG')]
    # Helper to copy images and labels as we go into the child
    def copy_img_and_label(img_name, parent_dir, child_dir, split):
      src_img = os.path.join(parent_dir, f'images/{split}', img_name)
      src_lbl = os.path.join(parent_dir, f'labels/{split}', img_name.rsplit('.', 1)[0] + '.txt')
      dst_img = os.path.join(child_dir, f'images/{split}', img_name)
      dst_lbl = os.path.join(child_dir, f'labels/{split}', img_name.rsplit('.', 1)[0] + '.txt')
      shutil.copy(src_img, dst_img)
      shutil.copy(src_lbl, dst_lbl)
    
    p1_imgs = get_images(p1_dir, split)
    p2_imgs = get_images(p2_dir, split)
    min_len = min(len(p1_imgs), len(p2_imgs))
    if min_len == 0:
      print("One of the parents in sp crossover is empty")
      return
    crossover_point = random.randint(1, min_len - 1)
    for img in p1_imgs[:crossover_point]:
      copy_img_and_label(img, p1_dir, child_dir, split)
    for img in p2_imgs[crossover_point:]:
      copy_img_and_label(img, p2_dir, child_dir, split)

  def mate(self, other, num):
    global indiv_num, new_root, mutation_probability
    new_pop_dir_name='new_pop'
    child = Individual.__new__(Individual)  # Avoid calling __init__

    # Set unique directory for the child
    # child.number = indiv_num
    child.number=num
    child.indiv_dir = os.path.join(new_pop_dir_name, f'indiv_{child.number}')
    os.makedirs(child.indiv_dir)
    child.dir_str = os.path.join(child.indiv_dir, 'data.yaml')
    os.makedirs(os.path.join(child.indiv_dir, 'images/train'))
    os.makedirs(os.path.join(child.indiv_dir, 'images/val'))
    os.makedirs(os.path.join(child.indiv_dir, 'labels/train'))
    os.makedirs(os.path.join(child.indiv_dir, 'labels/val'))

    # indiv_num += 1
    self.single_point_crossover(other.indiv_dir,child.indiv_dir,'train')
    self.single_point_crossover(other.indiv_dir,child.indiv_dir,'val')

    # Write data.yaml
    with open(child.dir_str, 'w') as f:
      f.write(f"path: {os.path.abspath(child.indiv_dir)}\n")
      f.write("train: images/train\n")
      f.write("val: images/val\n")
      f.write("names:\n")
      with open(os.path.join(source_dir, 'data.yaml'), 'r') as original_yaml:
        for line in original_yaml:
          if line.strip().startswith('0:') or line.strip().startswith('1:') or ':' in line.strip()[0:3]:
            f.write("  " + line)

    # Evaluate fitness
    child.calculate_fitness()
    return child

  def print(self):
    print(f"\nActive Indiv: "+str(self.indiv_dir) + " Fitness: " + str(self.fitness))

  def copy(self, source):
    self.fitness = source.fitness
    self.number=source.indiv_num
    self.indiv_dir = source.indiv_dir
    self.dir_str = source.dir_str

  def __str__(self):
    output = ""
    output += "Active Indiv: " + str(self.indiv_dir) + "\n"
    output += "Fitness: " + str(self.fitness) + "\n\n"
    return output

  def __lt__(self, other):
      return self.fitness < other.fitness



class Population():
  def __init__(self):
    global pop_size,new_root
    self.average_fitness=0
    self.best_fitness=0
    self.generation_num=0
    self.the_pop = []
    for i in range(0,pop_size):
      self.the_pop.append(Individual())
    self.elite_dir=os.path.join(new_root, 'elites')
    if os.path.exists(self.elite_dir):
      shutil.rmtree(self.elite_dir)
    os.makedirs(self.elite_dir)
    self.calculate_avg_fitness()
    self.calculate_best_fitness()
  
  def calculate_avg_fitness(self):
    global pop_size
    total_fitness = 0
    for i in range(0, pop_size):
      total_fitness += self.the_pop[i].fitness
    self.average_fitness = total_fitness/pop_size
  
  def calculate_best_fitness(self):
    global pop_size
    best = self.the_pop[0].fitness
    for i in range(1, pop_size):
      if self.the_pop[i].fitness > best:
        best = self.the_pop[i].fitness
    self.best_fitness=best

  def tournament(self):
    global tournament_size #,pop_size,elitism_num
    competitors = random.sample(self.the_pop, tournament_size)
    return max(competitors, key=lambda x: x.fitness)

  def make_offspring(self,num):
    parent_1=self.tournament()
    parent_2=self.tournament()
    child = parent_1.mate(parent_2,num=num)
    return child
  
  def populate(self, individual_1, individual_2):
    global pop_size, elitism_num
    # self.archive_elites()
    self.the_pop.sort(reverse=True)  # Sort by fitness (best first)
    new_pop_dir_name='new_pop'
    if os.path.exists(new_pop_dir_name):
      shutil.rmtree(new_pop_dir_name)
    os.makedirs(f'{new_pop_dir_name}')

    # Save the top individuals
    for i in range(elitism_num):
      elite = self.the_pop[i].copy()
      new_path = os.path.join(f'{new_pop_dir_name}', f'indiv_{i}')
      os.makedirs(f'{new_path}')
      if elite.indiv_dir != new_path:
        shutil.copytree(elite.indiv_dir, new_path)
        elite.indiv_dir = new_path
      self.the_pop[i] = elite  # Reassign for clarity

    # Overwrite the rest with new offspring
    for i in range(elitism_num, pop_size):
      # Make a child
      child=self.make_offspring(num=i)
      self.the_pop[i] = child

    # Delete old indivs directory
    shutil.rmtree(new_root)

    # Rename 'new_pop' to 'indivs'
    os.rename(new_pop_dir_name, new_root)

    # calculate new population statistics
    self.generation_num+=1
    self.calculate_avg_fitness()
    self.calculate_best_fitness()

  def print(self):
    global pop_size
    print(f"Average Fitness: {self.average_fitness}")
    print(f"Best Fitness: {self.best_fitness}")
    for i in range(0,pop_size):
      print(self.the_pop[i])
  
  # def archive_elites(self):
  #   global elitism_num, new_root
  #   # Sort individuals by fitness (highest first)
  #   sorted_inds = sorted(self.the_pop, key=lambda x: x.fitness, reverse=True)
  #   self.elites = sorted_inds[:elitism_num]
  #   if os.path.exists(self.elite_dir):
  #     shutil.rmtree(self.elite_dir)
  #   for idx, elite in enumerate(self.elites):
  #       cur_elite_dir = os.path.join(self.elite_dir, f'indiv_{elite.number}')
  #       os.makedirs(cur_elite_dir)
  #       shutil.copytree(elite.indiv_dir, self.elite_dir)

p1=Population()
p1.print()
# for i in range(0,1):
#   print(f"\n\n\n\n Onto indiv {i}")
#   temp=Individual()
#   print(temp)