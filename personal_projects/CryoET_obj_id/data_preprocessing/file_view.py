import zarr
import napari

zarr_data = zarr.open('../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/wbp.zarr', mode='r')

print(zarr_data.tree())

volume = zarr_data['0'][:]

# Visualize with Napari
viewer = napari.Viewer()
viewer.add_image(volume, name='Full Volume', colormap='gray')
napari.run()