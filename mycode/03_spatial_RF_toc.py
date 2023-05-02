from constants import *
import sys

sys.path.append(r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\mosaic_analysis')
import ml_utils
import pandas as pd

target = 'total_organic_carbon_%'
distance = 588127  # Distance for spatial crossvalidation in meters obtained from semivariogram

df = pd.read_csv(dataset)

# Feature selection
## all_features = features + river_features
## nodata_features = [feature for feature, nan_value in df[river_features].isnull().all().items() if nan_value]
## all_features = [feature for feature in all_features if feature not in nodata_features]
gdf_proj = ml_utils.clean_data(df=df, features=features, target=target, proj=proj)
cv = ml_utils.spatial_train_test_split(gdf_proj=gdf_proj, subset_pol=subset_pol, n_folds=50)
# feature_selection = ml_utils.boruta_fs(model=rf, df=df, features=features, target=target,
#                                        iter=1000, param_distributions=param_distributions,
#                                        cv=cv, hyper_parameter_iter=100)
#
# # Model
final_features = ['surface_nitrate', 'surface_calcite', 'surface_iron', 'surface_chla', 'surface_silicate',
                  'surface_phosphate',
                  'bottom_current', 'bottom_nitrate',  'bottom_silicate', 'bottom_temp', 'bottom_o2',
                  'stratification_index',
                  'dist_bottom_downcurrent_river_all', 'dist_downcurrent_bottom', 'distance_depth', ]

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

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
                                          output_dir=os.path.join(file_dir_project, 'data', 'prediction_maps',
                                                                  'toc', 'new_model_flow'),
                                          out_raster_name='toc',
                                          accuracy_metrics=accuracy_metrics,
                                          model_performance_file_name=os.path.join(file_dir_project, 'data',
                                                                                   'model_performance_toc.csv'),
                                          cv_file_name=os.path.join(file_dir_project, 'data',
                                                                    'hyperparameters_toc.csv'))
