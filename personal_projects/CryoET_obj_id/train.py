import os
import tensorflow as tf
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
num_epochs_per_dataset = 10
learning_rate = 1e-4

# Loop over experiments and datasets
for experiment in experiments:
    for dataset_name in datasets:
        print(f"Training on experiment: {experiment}, dataset: {dataset_name}")

        # Construct dataset paths
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
            augmentations=None,
            shuffle_buffer_size=100,
            particle_type_mapping=particle_type_mapping,
        )

        for patch, labels in train_dataset.take(1):
            print("Patch shape:", patch.shape)
            print("Labels shape:", labels.shape)  # Should match one-hot encoding
            break

        val_dataset = get_dataset(
            tomogram_paths=[train_tomogram_path],
            patch_size=patch_size,
            batch_size=batch_size,
            voxel_spacing=voxel_spacing,
            augmentations=None,  # No augmentations for validation data
            shuffle_buffer_size=100,
            particle_type_mapping=particle_type_mapping,
        )

        for patch, labels in val_dataset.take(1):
            print("Patch shape:", patch.shape)
            print("Labels shape:", labels.shape)  # Should match one-hot encoding
            break

        # Add caching and prefetching for performance
        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # Build and compile the model
        input_shape = patch_size + (1,)  # Add channel dimension
        model = build_3d_unet(input_shape)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Set up callbacks
        callbacks = [
            ModelCheckpoint(f'models/{experiment}_{dataset_name}_best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6, monitor='val_loss'),
            TensorBoard(log_dir=f'logs/{experiment}_{dataset_name}')
        ]

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=num_epochs_per_dataset,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        # Save the final model
        model.save(f'models/{experiment}_{dataset_name}_final_model.h5')

        # Optionally, save the training history
        import pickle
        with open(f'models/{experiment}_{dataset_name}_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        print(f"Training completed for experiment: {experiment}, dataset: {dataset_name}")
        print("-" * 50)
