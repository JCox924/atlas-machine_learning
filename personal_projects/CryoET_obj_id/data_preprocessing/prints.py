from utils import load_annotations

path = '../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr'


print(load_annotations(path, 10.000))

