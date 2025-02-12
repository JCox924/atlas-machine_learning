import os
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy

# Import your modules
from data_preprocessing.data_preprocessor import get_dataset, load_annotations
from data_preprocessing.augmentations import random_flip_3d, random_rotate_90
from models.model import build_3d_unet

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Enable mixed precision policy
# Enable mixed precision policy
set_global_policy('mixed_float16')
print("Mixed precision policy enabled.")

# Define paths and parameters
experiments = [
    'TS_5_4',
    'TS_6_4',
    'TS_6_6',
    'TS_69_2',
    'TS_73_6',
    # Add more experiments as needed
]

datasets = [
    'VoxelSpacing10.000/denoised.zarr',
    'VoxelSpacing10.000/wbp.zarr',
    'VoxelSpacing10.000/ctfdeconvolved.zarr',
    'VoxelSpacing10.000/isonetcorrected.zarr'
]

base_train_path = 'train/static/ExperimentRuns'
patch_size = (128, 128, 128)
batch_size = 1  # Adjust based on your GPU memory
voxel_spacing = [10.0, 10.0, 10.0]
augmentations = [random_flip_3d, random_rotate_90]
num_epochs_base = 10
num_epochs_finetune = 5
learning_rate = 1e-4

# Initialize a variable to track the base model
base_model = None

# Loop through datasets for transfer learning
for i, dataset_name in enumerate(datasets):
    print(f"Training on dataset: {dataset_name}")

    # Combine all experiments for the current dataset
    all_train_datasets = []
    all_val_datasets = []

    for experiment in experiments:
        train_tomogram_path = os.path.join(base_train_path, experiment, dataset_name)

        # Ensure the paths exist
        if not os.path.exists(train_tomogram_path):
            print(f"Training data not found at {train_tomogram_path}. Skipping.")
            continue

        # Load particle_type_mapping (assuming it's consistent across datasets)
        annotations, particle_type_mapping = load_annotations(train_tomogram_path, voxel_spacing)

        # Split annotations into training and validation sets
        train_annotations, val_annotations = train_test_split(
            annotations,
            test_size=0.2,  # 20% for validation
            random_state=42
        )

        # Prepare the datasets
        train_dataset = get_dataset(
            tomogram_paths=[train_tomogram_path],
            patch_size=patch_size,
            batch_size=batch_size,
            voxel_spacing=voxel_spacing,
            augmentations=augmentations,
            shuffle_buffer_size=100,
            particle_type_mapping=particle_type_mapping,
        )
        val_dataset = get_dataset(
            tomogram_paths=[train_tomogram_path],
            patch_size=patch_size,
            batch_size=batch_size,
            voxel_spacing=voxel_spacing,
            augmentations=None,  # No augmentations for validation data
            shuffle_buffer_size=100,
            particle_type_mapping=particle_type_mapping,
        )

        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)

    # Combine all datasets
    combined_train_dataset = tf.data.Dataset.sample_from_datasets(all_train_datasets).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    combined_val_dataset = tf.data.Dataset.sample_from_datasets(all_val_datasets).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build the model for the first dataset or load the pre-trained base model
    if base_model is None:
        print("Building the base model...")
        input_shape = patch_size + (1,)  # Add channel dimension
        base_model = build_3d_unet(input_shape)
        base_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        training_epochs = num_epochs_base
    else:
        print("Loading pre-trained base model and fine-tuning...")
        training_epochs = num_epochs_finetune

        # Freeze earlier layers
        for layer in base_model.layers[:-5]:  # Freeze all but the last few layers
            layer.trainable = False

        # Compile with a lower learning rate for fine-tuning
        base_model.compile(optimizer=Adam(learning_rate=learning_rate / 10), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(f'models/{dataset_name}_best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.1, patience=2, min_lr=1e-6, monitor='val_loss'),
        TensorBoard(log_dir=f'logs/{dataset_name}')
    ]

    # Train the model
    history = base_model.fit(
        combined_train_dataset,
        epochs=training_epochs,
        validation_data=combined_val_dataset,
        callbacks=callbacks
    )

    # Save the fine-tuned model
    base_model.save(f'models/{dataset_name}_final_model.h5')

    # Save the training history
    with open(f'models/{dataset_name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Fine-tuning completed for dataset: {dataset_name}")
    print("-" * 50)

# Save the fully trained base model
base_model.save('models/final_base_model.h5')
print("Training completed on all datasets.")

