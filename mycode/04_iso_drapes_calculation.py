import rasterio
import numpy as np
import os

toc_file = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps\toc' \
           r'\new_model_flow\mean_toc.tif'
tn_file = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps' \
          r'\tn\mean_tn.tif'
sa_file = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps' \
          r'\sa\mean_sa.tif'
c13_file = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps' \
           r'\c13\mean_c13.tif'
c14_file = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps' \
           r'\c14\mean_c14.tif'

variables = [{'file': toc_file, 'target': 'toc'},
             {'file': tn_file, 'target': 'tn'},
             {'file': sa_file, 'target': 'sa'},
             {'file': c13_file, 'target': 'c13'},
             {'file': c14_file, 'target': 'c14'}]


def normalize_raster(raster_file, new_min, new_max, output_file):
    assert all([raster_file.endswith('.tif'), output_file.endswith('.tif')])
    with rasterio.open(raster_file) as src:
        meta = src.meta
        data = src.read(1)
        data = np.ma.masked_where(data == src.nodata, data, copy=True)
    old_min = np.nanmin(data)
    old_max = np.nanmax(data)
    output_data = (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    # Write output file
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(output_data.astype(rasterio.float32), 1)
    return output_data


dir_out = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin\data\prediction_maps\isodrapes'

for variable in variables:
    print(f'Normalizing raster {variable["target"]}')
    variable['data'] = normalize_raster(raster_file=variable['file'], new_min=0, new_max=1,
                                        output_file=os.path.join(dir_out, variable['target'] + '.tif'))
    variable['data'] = variable['data'].filled(np.nan)

# Sci-kit learn
rasters = np.stack([variable['data'] for variable in variables], axis=2)
# Propagate NaN values
rasters = np.where(np.any(np.isnan(rasters), axis=2, keepdims=True), np.nan, rasters)
# Convert to 2D array
X = rasters[:, :, :5].reshape((rasters.shape[0] * rasters.shape[1], rasters.shape[2]))
X_nonans = X[~np.isnan(X).all(axis=1)]
X = np.nan_to_num(X, nan=-9999, posinf=-9999, neginf=-9999)

# Perform k-means cluster
from sklearn import cluster
print('Determining best number of clusters')
inertias = []
n_clusters = np.arange(start=2, stop=10, step=1)
for n_cluster in n_clusters:
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_cluster)
    kmeans.fit(X_nonans)
    cluster_labels = kmeans.labels_
    inertias.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(n_clusters, inertias)
plt.savefig('kmeans_n_clusters.jpg')

print(f'Calculating k-means')
kmeans = cluster.MiniBatchKMeans(n_clusters=4)
kmeans.fit(X)
isodrapes = kmeans.labels_
isodrapes = isodrapes.reshape(rasters[:, :, 0].shape)
isodrapes = isodrapes.astype(int)
# Reshape back the variables_fig1
X = X.reshape(rasters[:, :].shape)
# Extract 1 variable to mask the ISODRAPES raster
X = X[:, :, 0]
# Mask ISODRAPES raster
isodrapes = np.where(X == -9999, np.nan, isodrapes)
# # # Modify order of isodrape clusters
isodrapes = np.where(isodrapes == 2, 4, isodrapes)
isodrapes = np.where(isodrapes == 1, 2, isodrapes)
isodrapes = np.where(isodrapes == 4, 1, isodrapes)

# Save raster
print(f'Saving ISODRAPES raster')
output_file = os.path.join(dir_out, 'isodrapes.tif')
with rasterio.open(variables[0]['file']) as src:
    meta = src.meta
with rasterio.open(output_file, 'w', **meta) as dst:
    dst.write(isodrapes, 1)

