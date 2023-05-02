from constants import *
import sys

sys.path.append(r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\mosaic_analysis')
import ml_utils
import pandas as pd

target = 'surface_area_m2_g'

df = pd.read_csv(dataset)

# Feature selection
gdf_proj = ml_utils.clean_data(df=df, features=features, target=target, proj=proj)
# cv = ml_utils.spatial_train_test_split(gdf_proj=gdf_proj, subset_pol=subset_pol, n_folds=50)
# feature_selection = ml_utils.boruta_fs(model=rf, df=df, features=features, target=target,
#                                        iter=1000, param_distributions=param_distributions,
#                                        cv=cv, hyper_parameter_iter=100)

final_features = ['bottom_sal', 'bathymetry', 'dist_downcurrent_bottom', 'bottom_temp', 'surface_calcite',
                  'surface_silicate', 'distance_depth', 'surface_nitrate', 'surface_iron']

# Model
cv_method = ml_utils.spatial_train_test_split
cv_method_dict = dict(subset_pol=subset_pol, n_folds=50, test_size=0.2)

ml_utils.nested_cross_validation_modeling(gdf_proj=gdf_proj,
                                          features=final_features,
                                          target=target,
                                          model=rf,
                                          cv_method_inner=cv_method,
                                          cv_method_inner_dict=cv_method_dict,
                                          cv_method_outer=cv_method,
                                          cv_method_outer_dict=cv_method_dict,
                                          n_iter=500,
                                          param_distributions=param_distributions,
                                          file_dir_rasters=file_dir,
                                          output_dir=os.path.join(file_dir_project, 'data', 'prediction_maps', 'sa'),
                                          out_raster_name='sa',
                                          accuracy_metrics=accuracy_metrics,
                                          model_performance_file_name=os.path.join(file_dir_project, 'data',
                                                                                   'model_performance_sa.csv'),
                                          hyperparameters_file_name=os.path.join(file_dir_project, 'data',
                                                                                 'hyperparameters_sa.csv'))