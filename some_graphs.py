


"""
import matplotlib.pyplot as plt

# Data
train_sizes = [107, 148, 448]
epochs = [156, 177, 50]
training_times_hours = [0.027, 0.050, 0.132]

# Convert training times to minutes
training_times_minutes = [t * 60 for t in training_times_hours]

# Calculate total images trained
# total_images_trained = [size * epoch for size, epoch in zip(train_sizes, epochs)]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, training_times_minutes, marker='o', linestyle='-', color='blue')

# Annotate each point with total images trained
for size, time, train_size in zip(train_sizes, training_times_minutes, train_sizes):
    plt.annotate(f'{train_size} imgs trained', (size, time), textcoords="offset points", xytext=(0,10), ha='center')

# Labels and title
plt.title('Training Size vs Training Time')
plt.xlabel('Number of Training Images')
plt.ylabel('Training Time (minutes)')
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('training_time_vs_size_minutes.png', dpi=300)
# plt.show()  # Uncomment to display the plot

"""



import matplotlib.pyplot as plt

# Data
train_sizes = [107, 148, 448]
epochs = [156, 177, 50]
training_times_hours = [0.027, 0.050, 0.132]
training_times_minutes = [round(t * 60, 2) for t in training_times_hours]  # rounded for table clarity

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

# --- Plot on the top subplot ---
axs[0].plot(train_sizes, training_times_minutes, marker='o', linestyle='-', color='blue')

# Annotate each point
for size, time in zip(train_sizes, training_times_minutes):
    axs[0].annotate(f'{size} imgs trained', (size, time), textcoords="offset points", xytext=(0,10), ha='center')

axs[0].set_title('Training Size vs Training Time')
axs[0].set_xlabel('Number of Training Images')
axs[0].set_ylabel('Training Time (minutes)')
axs[0].grid(True)

# --- Table on the bottom subplot ---
# Prepare table data
column_labels = ["Training Images", "Epochs", "Training Time (min)"]
table_data = list(zip(train_sizes, epochs, training_times_minutes))

# Create the table
axs[1].axis('off')  # Hide the plot area for the table
table = axs[1].table(cellText=table_data, colLabels=column_labels, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Save the combined figure
plt.tight_layout()
plt.savefig('training_plot_with_table.png', dpi=300)
# plt.show()  # Uncomment this line to display the figure
