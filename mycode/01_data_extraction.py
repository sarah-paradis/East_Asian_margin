import os
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from scipy import stats
from constants import *
import sys

sys.path.insert(0, r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_Server')
from mosaic import mosaic

# Extract data from MOSAIC
material_analyzed_variables = {'bulk': ['total_organic_carbon_%', 'total_nitrogen_%', 'OC:TN_ratio',
                                        'mean_grain_size_microm', 'surface_area_m2_g', 'mud_silt_clay_%',
                                        'sand_%', 'gravel_%', 'silt_%', 'clay_%_2microm', 'clay_%_4microm'],
                               'TOC': ['Fm_14C', 'Delta_13C', 'Delta_14C']}
metadata = ['core_id', 'sample_id', 'sample_name', 'sample_depth_upper_cm', 'sample_depth_bottom_cm',
            'sample_depth_average_cm', 'sample_comment', 'latitude', 'longitude', 'georeferenced_coordinates',
            'sampling_date', 'sampling_year', 'water_depth_m', 'core_length_cm', 'longhurst_provinces_full',
            'exclusive_economics_zone', 'seas', 'marcats', 'core_comment', 'research_vessel',
            'sampling_campaign_name', 'sampling_campaign_date_start', 'sampling_campaign_date_end',
            'sampling_method_type', 'water_depth_source_desc', 'replicate']
df = pd.DataFrame()
for material_analyzed, variables in material_analyzed_variables.items():
    df_temp = mosaic.sample_analyses(variable=variables, general_variables=metadata,
                                     sample_depths=(0, 5), sample_depth_averages=(0, 3),
                                     material_analyzed=material_analyzed, calculated=True, published=False)
    if df.empty:
        df = df_temp.copy()
    else:
        df = pd.merge(right=df, left=df_temp, on=metadata, how='outer')

# Convert dataframe into a Geodataframe
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=crs)

# Open polygon
print('Opening polygon of study area')
pol = gpd.read_file(os.path.join(file_dir, study_area + '.shp'))

# Extract data within that polygon
print('Clipping data to the polygon')
# First make sure that they are in the same projection
if pol.crs != gdf.crs:
    gdf = gdf.to_crs(pol.crs)
assert pol.crs == gdf.crs
gdf_EA = gpd.clip(gdf, pol)
del gdf


# Calculate Delta 14C
def delta_14c(fm, yc):
    return (fm * np.exp(1 / 8267 * (1950 - yc)) - 1) * 1000


conditions_c14 = [(gdf_EA['Delta_14C'].isnull()) & (gdf_EA['Fm_14C'].notnull()),
                  (gdf_EA['Delta_14C']).notnull()]
choices_c14 = [delta_14c(fm=gdf_EA['Fm_14C'], yc=gdf_EA['sampling_year'].fillna(2014)),
               gdf_EA['Delta_14C']]

gdf_EA['Delta_14C'] = np.select(conditions_c14, choices_c14, default=np.nan)

# Extract TN from OC/TN
gdf_EA['total_nitrogen_%'].fillna(gdf_EA['total_organic_carbon_%']/gdf_EA['OC:TN_ratio']*12/14, inplace=True)

# Harmonize grain size to surface area
conditions = [(gdf_EA['mud_silt_clay_%'].isnull()) & (gdf_EA['silt_%'].notnull()) &
              (gdf_EA['clay_%_4microm'].notnull()),
              (gdf_EA['mud_silt_clay_%'].isnull()) & (gdf_EA['sand_%'].notnull()),
              gdf_EA['mud_silt_clay_%'].notnull()]
choices = [gdf_EA['silt_%'] + gdf_EA['clay_%_4microm'],
           1 - gdf_EA['sand_%'],
           gdf_EA['mud_silt_clay_%']]

gdf_EA['mud_harmonized'] = np.select(conditions, choices, default=np.nan)

df_nonans_mgs_sa = gdf_EA[['mean_grain_size_microm', 'surface_area_m2_g']].dropna()
df_nonans_mud_sa = gdf_EA[['mud_harmonized', 'surface_area_m2_g']].replace(0, np.nan).dropna()

mgs_1 = df_nonans_mgs_sa['mean_grain_size_microm']
sa_1 = df_nonans_mgs_sa['surface_area_m2_g']
df_nonans_mgs = gdf_EA.dropna(subset=['mean_grain_size_microm'])
sa_to_predict_from_mgs = df_nonans_mgs[df_nonans_mgs['surface_area_m2_g'].isnull()]['mean_grain_size_microm']

mud_2 = df_nonans_mud_sa['mud_harmonized']
sa_2 = df_nonans_mud_sa['surface_area_m2_g']
df_nonans_mud = gdf_EA.replace(0, np.nan).dropna(subset=['mud_harmonized'])
sa_to_predict_from_mud = df_nonans_mud[df_nonans_mud['surface_area_m2_g'].isnull()]['mud_harmonized']

# Transform data
transformer = FunctionTransformer(np.log10, validate=True)

mgs_trans_1 = transformer.fit_transform(mgs_1.values.reshape(-1, 1))
sa_trans_1 = transformer.fit_transform(sa_1.values.reshape(-1, 1))
sa_to_predict_from_mgs_trans_1 = transformer.fit_transform(sa_to_predict_from_mgs.values.reshape(-1, 1))

mud_trans_2 = mud_2.values.reshape(-1, 1)
sa_trans_2 = transformer.fit_transform(sa_2.values.reshape(-1, 1))
sa_to_predict_from_mud_trans_1 = sa_to_predict_from_mud.values.reshape(-1, 1)

# Regression
from sklearn.model_selection import cross_validate, ShuffleSplit, learning_curve

regressor_1 = LinearRegression()
cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results = cross_validate(regressor_1, mgs_trans_1, sa_trans_1, cv=cv, scoring="neg_mean_absolute_percentage_error")

# Regression
regressor_1 = LinearRegression(fit_intercept=True)
mgs_test_1 = np.linspace(mgs_trans_1.min(), mgs_trans_1.max(), 500)[:, None]
cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results_1 = cross_validate(regressor_1, mgs_trans_1, sa_trans_1, cv=cv, scoring="neg_mean_absolute_error")
regressor_1.fit(mgs_trans_1, sa_trans_1)
print(f"The R^2 from mean grain size is {regressor_1.score(mgs_trans_1, sa_trans_1):.2f}")
cv_result_1 = cross_validate(regressor_1, mgs_trans_1, sa_trans_1, cv=10)
print(f"The mean absolute error from mean grain size is "
      f"{-cv_results_1['test_score'].mean():.2f} +/- {cv_results_1['test_score'].std():.2f}")
sa_mgs_curve = regressor_1.predict(mgs_test_1)
sa_mgs_fit = regressor_1.predict(sa_to_predict_from_mgs_trans_1)
intercept_1 = 10 ** regressor_1.intercept_[0]
coef_1 = regressor_1.coef_[0][0]
assert np.round(coef_1, 2) == -0.39
function_1 = f'SA = {intercept_1:.2f} x MGS$^{{-0.39}}$'
r2_1 = f'{regressor_1.score(mgs_trans_1, sa_trans_1):.2f}'
MAE_1 = f"{10 ** -cv_results_1['test_score'].mean():.2f} \u00B1 {10 ** cv_results_1['test_score'].std():.2f}"

regressor_2 = LinearRegression()
mud_test_2 = np.linspace(mud_trans_2.min(), mud_trans_2.max(), 100)[:, None]
cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results_2 = cross_validate(regressor_2, mud_trans_2, sa_trans_2, cv=cv, scoring="neg_mean_absolute_error")
regressor_2.fit(mud_trans_2, sa_trans_2)

print(f"The R^2 from mud content is {regressor_2.score(mud_trans_2, sa_trans_2):.2f}")
cv_result_2 = cross_validate(regressor_2, mud_trans_2, sa_trans_2, cv=10)
print(f"The mean absolute error from mud content is "
      f"{10 ** (-cv_results_2['test_score'].mean()):.2f} +/- {10 ** cv_results_2['test_score'].std():.2f}")
sa_mud_curve = regressor_2.predict(mud_test_2)
sa_mud_fit = regressor_2.predict(sa_to_predict_from_mud_trans_1)

function_2 = f'SA = {regressor_2.intercept_[0]:.2f} x {regressor_2.coef_[0][0]:.2f}$^{{mud}}$'
r2_2 = f'{regressor_2.score(mud_trans_2, sa_trans_2):.2f}'
MAE_2 = f"{10 ** (-cv_results_2['test_score'].mean()):.2f} \u00B1 {10 ** cv_results_2['test_score'].std():.2f}"

# Visualization
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), constrained_layout=True)
# Plot MGS to Surface area
n_observed = len(mgs_1)
n_predicted = len(sa_to_predict_from_mgs)
axs[0].scatter(mgs_1, sa_1, color='cornflowerblue', edgecolor='white',
               label=f'Observed (n={n_observed})', s=50)
axs[0].scatter(sa_to_predict_from_mgs, 10 ** (sa_mgs_fit),
               color='palevioletred', edgecolor='white', s=75,
               label=f'Predicted (n={n_predicted})')
# axs[0].plot(np.exp(mgs_test_1), np.exp(sa_mgs_curve), "k--", label="Fit")
axs[0].annotate(f'{function_1}', xy=(0.45, 0.775), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[0].annotate(f'R\u00b2 = {r2_1}', xy=(0.45, 0.7), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[0].annotate(f'MAE = {MAE_1}', xy=(0.45, 0.65), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[0].annotate(f'p < 0.001', xy=(0.45, 0.6), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[0].set_xlabel('Mean grain size ($\mathregular{\mu}$m)', fontsize=16)
axs[0].set_xlim(0, 800)
# Plot Mud to Surface area
n_observed = len(mud_2)
n_predicted = len(sa_to_predict_from_mud)
axs[1].scatter(mud_2, sa_2, color='cornflowerblue', edgecolor='white',
               label=f'Observed (n={n_observed})', s=50)
axs[1].scatter(sa_to_predict_from_mud, 10 ** (sa_mud_fit),
               color='palevioletred', edgecolor='white', s=75,
               label=f'Predicted (n={n_predicted})')
axs[1].annotate(f'{function_2}', xy=(0.45, 0.775), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[1].annotate(f'R\u00b2 = {r2_2}', xy=(0.45, 0.7), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[1].annotate(f'MAE = {MAE_2}', xy=(0.45, 0.65), xycoords='axes fraction', va='center', ha='left', fontsize=12)
axs[1].annotate(f'p < 0.001', xy=(0.45, 0.6), xycoords='axes fraction', va='center', ha='left', fontsize=12)
# axs[1].plot(mud_test_2, np.exp(sa_mud_curve), "k--", label="Fit")
for ax in axs:
    ax.tick_params('both', which='major', labelsize=12)
    ax.set_ylabel('Sediment surface area (m$^2$·g$^{-2}$)', fontsize=16)
    ax.set_ylim(0, 50)
axs[1].set_xlabel('Mud content (%)', fontsize=16)
axs[1].set_xlim(0, 102)
axs[0].legend(fontsize=12)
axs[1].legend(fontsize=12)
axs[0].annotate('a)', xy=(0.01, 0.95), xycoords='axes fraction', fontsize=14, weight='bold')
axs[1].annotate('b)', xy=(0.01, 0.95), xycoords='axes fraction', fontsize=14, weight='bold')
plt.savefig(os.path.join(file_dir_project, 'figures', 'S1_SA_data_harmonization.jpg'), dpi=300)

# Apply this harmonization to the dataset
print('Applying SA harmonization to the dataset')
pd.options.mode.chained_assignment = None  # default='warn'

target = np.log10(gdf_EA['mean_grain_size_microm'].dropna()).to_numpy().reshape(-1, 1)
index = gdf_EA['mean_grain_size_microm'].dropna().index
gdf_EA['SA_from_mgs'] = np.nan
gdf_EA['SA_from_mgs'].loc[index] = 10 ** (regressor_1.predict(target).flatten())

target = gdf_EA['mud_harmonized'].replace(0, np.nan).dropna().to_numpy().reshape(-1, 1)
index = gdf_EA['mud_harmonized'].replace(0, np.nan).dropna().index
gdf_EA['SA_from_mud'] = np.nan
gdf_EA['SA_from_mud'].loc[index] = 10 ** (regressor_2.predict(target).flatten())

conditions = [(gdf_EA['surface_area_m2_g'].isnull()) & (gdf_EA['SA_from_mud'].notnull()),
              (gdf_EA['surface_area_m2_g'].isnull()) & (gdf_EA['SA_from_mud'].isnull()) &
              (gdf_EA['SA_from_mgs'].notnull()),
              gdf_EA['surface_area_m2_g'].notnull()]
choices = [gdf_EA['SA_from_mud'],
           gdf_EA['SA_from_mgs'],
           gdf_EA['surface_area_m2_g']]

gdf_EA['SA_harmonized'] = np.select(conditions, choices, default=np.nan)

# Group by data per core_id and average it
gdf_EA_grouped = gdf_EA.groupby('core_id').mean()
gdf_EA_grouped_first = gdf_EA.groupby('core_id').first()

# Save output
# gdf_EA_grouped.to_csv(dataset, index=False)
gdf_EA_grouped_first.to_csv(os.path.join(file_dir, 'E_Asian_data_first.csv'), index=False)
