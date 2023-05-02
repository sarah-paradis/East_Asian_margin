from constants import *
import sys

sys.path.append(r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\mosaic_analysis')
import ml_utils
import pandas as pd


target = 'Delta_13C'

df = pd.read_csv(dataset)

# Feature selection
gdf_proj = ml_utils.clean_data(df=df, features=features, target=target, proj=proj)
cv = ml_utils.spatial_train_test_split(gdf_proj=gdf_proj, subset_pol=subset_pol, n_folds=50)
feature_selection = ml_utils.boruta_fs(model=rf, df=df, features=features, target=target,
                                       iter=1000, param_distributions=param_distributions,
                                       cv=cv, hyper_parameter_iter=100)

final_features = ['dist_bottom_downcurrent_river_all', 'distalloc_bottom_downcurrent_river_all',
                  'surface_phosphate', 'surface_silicate', 'surface_chla', 'surface_iron',
                  'bottom_current', 'bottom_o2', 'bottom_chla', 'bottom_temp',
                  'distance_depth', 'bottom_trawling_all', 'dist_downcurrent_bottom']

# Model
