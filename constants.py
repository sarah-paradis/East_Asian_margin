import cartopy.crs as ccrs
import numpy as np
import rasterio
import cartopy.feature as cfeature
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys

sys.path.append(r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\mosaic_analysis')
from figures_utils import scale_bar
from ml_utils import loguniform_int
from scipy.stats import loguniform
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

file_dir = r'F:\Data\East_Asian_marginal_seas'
file_dir_project = r'C:\Users\sparadis\Documents\ETH\MOSAIC\MOSAIC_data_analysis\E_Asian_margin'
projection = ccrs.AzimuthalEquidistant(central_longitude=116.2724444444444, central_latitude=21.05894444444444)
study_area = 'E_Asian_seas_4'
crs = 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (' \
      'Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],' \
      'MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World ' \
      'Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,' \
      '298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",' \
      '0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",' \
      '0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],' \
      'USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]] '

proj = 'PROJCS["E_Asian_equidistant_projection",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",116.2724444444444],PARAMETER["Latitude_Of_Origin",21.05894444444444],UNIT["Meter",1.0]]'
pol_seas = os.path.join(file_dir, 'E_Asian_class_2.shp')
subset_pol = os.path.join(file_dir, f'subset_polygons_4.shp')

dataset = os.path.join(file_dir, 'E_Asian_data.csv')

# River features
river_features = ['dist_euc_river_planar_Changhuajiang', 'dist_surface_downcurrent_river_Changhuajiang',
                  'dist_bottom_downcurrent_river_Changhuajiang', 'dist_euc_river_planar_Changjiang',
                  'dist_surface_downcurrent_river_Changjiang', 'dist_bottom_downcurrent_river_Changjiang',
                  'dist_euc_river_planar_Dalinghe', 'dist_surface_downcurrent_river_Dalinghe',
                  'dist_bottom_downcurrent_river_Dalinghe', 'dist_euc_river_planar_Guanhe',
                  'dist_surface_downcurrent_river_Guanhe', 'dist_bottom_downcurrent_river_Guanhe',
                  'dist_euc_river_planar_Haihe', 'dist_surface_downcurrent_river_Haihe',
                  'dist_bottom_downcurrent_river_Haihe', 'dist_euc_river_planar_Hanjiang',
                  'dist_surface_downcurrent_river_Hanjiang', 'dist_bottom_downcurrent_river_Hanjiang',
                  'dist_euc_river_planar_Huanghe', 'dist_surface_downcurrent_river_Huanghe',
                  'dist_bottom_downcurrent_river_Huanghe', 'dist_euc_river_planar_Jinjiang',
                  'dist_surface_downcurrent_river_Jinjiang', 'dist_bottom_downcurrent_river_Jinjiang',
                  'dist_euc_river_planar_Jiulongjiang', 'dist_surface_downcurrent_river_Jiulongjiang',
                  'dist_bottom_downcurrent_river_Jiulongjiang', 'dist_euc_river_planar_Liaohe',
                  'dist_surface_downcurrent_river_Liaohe', 'dist_bottom_downcurrent_river_Liaohe',
                  'dist_euc_river_planar_Luanhe', 'dist_surface_downcurrent_river_Luanhe',
                  'dist_bottom_downcurrent_river_Luanhe', 'dist_euc_river_planar_Majiahe',
                  'dist_surface_downcurrent_river_Majiahe', 'dist_bottom_downcurrent_river_Majiahe',
                  'dist_euc_river_planar_Minjiang', 'dist_surface_downcurrent_river_Minjiang',
                  'dist_bottom_downcurrent_river_Minjiang', 'dist_euc_river_planar_Moyangjiang',
                  'dist_surface_downcurrent_river_Moyangjiang', 'dist_bottom_downcurrent_river_Moyangjiang',
                  'dist_euc_river_planar_Nandujiang', 'dist_surface_downcurrent_river_Nandujiang',
                  'dist_bottom_downcurrent_river_Nandujiang', 'dist_euc_river_planar_Nanliujiang',
                  'dist_surface_downcurrent_river_Nanliujiang', 'dist_bottom_downcurrent_river_Nanliujiang',
                  'dist_euc_river_planar_Oujiang', 'dist_surface_downcurrent_river_Oujiang',
                  'dist_bottom_downcurrent_river_Oujiang', 'dist_euc_river_planar_Qiantangjiang',
                  'dist_surface_downcurrent_river_Qiantangjiang', 'dist_bottom_downcurrent_river_Qiantangjiang',
                  'dist_euc_river_planar_Tuhaihe', 'dist_surface_downcurrent_river_Tuhaihe',
                  'dist_bottom_downcurrent_river_Tuhaihe', 'dist_euc_river_planar_Yalujiang',
                  'dist_surface_downcurrent_river_Yalujiang', 'dist_bottom_downcurrent_river_Yalujiang',
                  'dist_euc_river_planar_Zhujiang', 'dist_surface_downcurrent_river_Zhujiang',
                  'dist_bottom_downcurrent_river_Zhujiang', 'dist_euc_river_planar_Han',
                  'dist_surface_downcurrent_river_Han', 'dist_bottom_downcurrent_river_Han',
                  'dist_euc_river_planar_Keum', 'dist_surface_downcurrent_river_Keum',
                  'dist_bottom_downcurrent_river_Keum', 'dist_euc_river_planar_Mankyong',
                  'dist_surface_downcurrent_river_Mankyong', 'dist_bottom_downcurrent_river_Mankyong',
                  'dist_euc_river_planar_Sapgyo', 'dist_surface_downcurrent_river_Sapgyo',
                  'dist_bottom_downcurrent_river_Sapgyo', 'dist_euc_river_planar_Seumjin',
                  'dist_surface_downcurrent_river_Seumjin', 'dist_bottom_downcurrent_river_Seumjin',
                  'dist_euc_river_planar_Yeongsan', 'dist_surface_downcurrent_river_Yeongsan',
                  'dist_bottom_downcurrent_river_Yeongsan', 'dist_euc_river_planar_Kelantan',
                  'dist_surface_downcurrent_river_Kelantan', 'dist_bottom_downcurrent_river_Kelantan',
                  'dist_euc_river_planar_Pahang', 'dist_surface_downcurrent_river_Pahang',
                  'dist_bottom_downcurrent_river_Pahang', 'dist_euc_river_planar_Rajang',
                  'dist_surface_downcurrent_river_Rajang', 'dist_bottom_downcurrent_river_Rajang',
                  'dist_euc_river_planar_Trengganu', 'dist_surface_downcurrent_river_Trengganu',
                  'dist_bottom_downcurrent_river_Trengganu', 'dist_euc_river_planar_Chishui',
                  'dist_surface_downcurrent_river_Chishui', 'dist_bottom_downcurrent_river_Chishui',
                  'dist_euc_river_planar_Choshui', 'dist_surface_downcurrent_river_Choshui',
                  'dist_bottom_downcurrent_river_Choshui', 'dist_euc_river_planar_Erhjen',
                  'dist_surface_downcurrent_river_Erhjen', 'dist_bottom_downcurrent_river_Erhjen',
                  'dist_euc_river_planar_Hoping', 'dist_surface_downcurrent_river_Hoping',
                  'dist_bottom_downcurrent_river_Hoping', 'dist_euc_river_planar_Houlung',
                  'dist_surface_downcurrent_river_Houlung', 'dist_bottom_downcurrent_river_Houlung',
                  'dist_euc_river_planar_Hsiukuluan', 'dist_surface_downcurrent_river_Hsiukuluan',
                  'dist_bottom_downcurrent_river_Hsiukuluan', 'dist_euc_river_planar_Hualien',
                  'dist_surface_downcurrent_river_Hualien', 'dist_bottom_downcurrent_river_Hualien',
                  'dist_euc_river_planar_Kaoping', 'dist_surface_downcurrent_river_Kaoping',
                  'dist_bottom_downcurrent_river_Kaoping', 'dist_euc_river_planar_Lanyang',
                  'dist_surface_downcurrent_river_Lanyang', 'dist_bottom_downcurrent_river_Lanyang',
                  'dist_euc_river_planar_Linpien', 'dist_surface_downcurrent_river_Linpien',
                  'dist_bottom_downcurrent_river_Linpien', 'dist_euc_river_planar_Pachang',
                  'dist_surface_downcurrent_river_Pachang', 'dist_bottom_downcurrent_river_Pachang',
                  'dist_euc_river_planar_Peikang', 'dist_surface_downcurrent_river_Peikang',
                  'dist_bottom_downcurrent_river_Peikang', 'dist_euc_river_planar_Peinan',
                  'dist_surface_downcurrent_river_Peinan', 'dist_bottom_downcurrent_river_Peinan',
                  'dist_euc_river_planar_Potzu', 'dist_surface_downcurrent_river_Potzu',
                  'dist_bottom_downcurrent_river_Potzu', 'dist_euc_river_planar_Taan',
                  'dist_surface_downcurrent_river_Taan', 'dist_bottom_downcurrent_river_Taan',
                  'dist_euc_river_planar_Tachia', 'dist_surface_downcurrent_river_Tachia',
                  'dist_bottom_downcurrent_river_Tachia', 'dist_euc_river_planar_Tanshui',
                  'dist_surface_downcurrent_river_Tanshui', 'dist_bottom_downcurrent_river_Tanshui',
                  'dist_euc_river_planar_Touchien', 'dist_surface_downcurrent_river_Touchien',
                  'dist_bottom_downcurrent_river_Touchien', 'dist_euc_river_planar_Tsengwen',
                  'dist_surface_downcurrent_river_Tsengwen', 'dist_bottom_downcurrent_river_Tsengwen',
                  'dist_euc_river_planar_Tungkang', 'dist_surface_downcurrent_river_Tungkang',
                  'dist_bottom_downcurrent_river_Tungkang', 'dist_euc_river_planar_Yenshui',
                  'dist_surface_downcurrent_river_Yenshui', 'dist_bottom_downcurrent_river_Yenshui',
                  'dist_euc_river_planar_Khlong_Phum_Duang', 'dist_surface_downcurrent_river_Khlong_Phum_Duang',
                  'dist_bottom_downcurrent_river_Khlong_Phum_Duang', 'dist_euc_river_planar_Mae_Klong',
                  'dist_surface_downcurrent_river_Mae_Klong', 'dist_bottom_downcurrent_river_Mae_Klong',
                  'dist_euc_river_planar_Pattani', 'dist_surface_downcurrent_river_Pattani',
                  'dist_bottom_downcurrent_river_Pattani', 'dist_euc_river_planar_Ba',
                  'dist_surface_downcurrent_river_Ba', 'dist_bottom_downcurrent_river_Ba', 'dist_euc_river_planar_Ma',
                  'dist_surface_downcurrent_river_Ma', 'dist_bottom_downcurrent_river_Ma',
                  'dist_euc_river_planar_Mekong', 'dist_surface_downcurrent_river_Mekong',
                  'dist_bottom_downcurrent_river_Mekong', 'dist_euc_river_planar_Sai_Gon',
                  'dist_surface_downcurrent_river_Sai_Gon', 'dist_bottom_downcurrent_river_Sai_Gon',
                  'dist_euc_river_planar_Song_Hong', 'dist_surface_downcurrent_river_Song_Hong',
                  'dist_bottom_downcurrent_river_Song_Hong', 'dist_euc_river_planar_Thai_Binh',
                  'dist_surface_downcurrent_river_Thai_Binh', 'dist_bottom_downcurrent_river_Thai_Binh',
                  'dist_euc_river_planar_Thu_Bon', 'dist_surface_downcurrent_river_Thu_Bon',
                  'dist_bottom_downcurrent_river_Thu_Bon']
# Only non-river features
features = ['bathymetry', 'slope', 'aspect', 'mean_curvature',
            'bottom_chla', 'bottom_current', 'bottom_nitrate', 'bottom_o2', 'bottom_phosphate',
            'bottom_sal', 'bottom_silicate', 'bottom_temp',
            'surface_calcite', 'surface_chla',
            'surface_iron', 'surface_nitrate', 'surface_npp',
            'surface_phosphate', 'surface_silicate',
            'distance_depth',
            'dist_euc_planar', 'dist_surface', 'dist_downcurrent_bottom',
            'dist_euc_river_planar_all', 'dist_bottom_downcurrent_river_all',
            'distalloc_bottom_downcurrent_river_all', 'bottom_trawling_all', 'stratification_index',
            # 'dist_surface_downcurrent_river_all', 'dist_downcurrent_surface', 'bottom_npp',
            # 'distalloc_euc_river_planar_all', 'distalloc_surface_downcurrent_river_all', 'significant_wave_height'
            ]

# Hyperparameters that need tuning
param_distributions = {'n_estimators': loguniform_int(100, 600),
                       'max_depth': loguniform_int(10, 70),
                       'max_features': loguniform(0.1, 1),
                       'max_samples': loguniform(0.4, 1),
                       'min_samples_split': loguniform_int(2, 20),
                       'max_leaf_nodes': loguniform_int(40, 700),
                       'min_samples_leaf': loguniform_int(1, 8)}
param_distributions_pipeline = {'sfs__estimator__n_estimators': loguniform_int(100, 500),
                                'sfs__estimator__max_depth': loguniform_int(10, 40),
                                'sfs__estimator__max_features': loguniform(0.2, 1),
                                'sfs__estimator__max_samples': loguniform(0.1, 1),
                                'sfs__estimator__min_samples_split': loguniform_int(2, 20),
                                'sfs__estimator__max_leaf_nodes': loguniform_int(20, 100),
                                'sfs__estimator__min_samples_leaf': loguniform_int(1, 20)}

rf = RandomForestRegressor(criterion='squared_error', oob_score=True, n_jobs=4)

accuracy_metrics = {'r2': r2_score, 'MAE': mean_absolute_error, 'MSE': mean_squared_error}


def general_map(ax, fig, bathymetry=False):
    """Function to plot general properties in the map:
    Coastline, Bathymetry, Scale, Gridlines etc."""
    # Plot main map of study area
    borders = cfeature.NaturalEarthFeature(category='cultural',
                                           name='admin_0_boundary_lines_land',
                                           scale='10m',
                                           facecolor='none')
    ax.add_feature(borders, edgecolor='black', lw=0.5)
    ax.add_feature(cfeature.LAND, facecolor='0.8')
    ax.coastlines(resolution='10m', color='black', lw=0.5)
    # Plot raster
    if bathymetry:
        with rasterio.open(os.path.join(file_dir, 'bathymetry.tif'), 'r') as src:
            raster_crs = src.crs.to_dict()
            left, bottom, right, top = src.bounds
            data = src.read()[0]
            data = np.ma.masked_where(data == src.nodata, data, copy=True)
            vmin = np.nanmin(data)
            vmax = 0
            vmean = np.nanmean(data)
            print(vmin)
        im = ax.imshow(data, transform=projection, cmap='Blues_r',
                       extent=(left, right, bottom, top), alpha=0.7,
                       vmin=vmin, vmax=vmax)
        cax = fig.add_axes([0.665, 0.125, 0.05, 0.15])
        plt.colorbar(im, cax=cax, orientation='vertical')
        cax.set_title('Water depth\n(m)', x=1.3, y=1.025)
        ax.add_patch(Rectangle(xy=[3.725*10**5, -1.82*10**6], width=7*10**5, height=10.5*10**5, edgecolor='k', lw=0.5,
                               fill=True, facecolor='white', alpha=0.85, zorder=10))
    ax.set_extent([98, 130, 0, 41])

    # Set gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'rotation': 0}
    gl.ylabel_style = {'size': 12, 'rotation': 0}
    #    ax.set_xticks([100, 110, 120, 130], crs=ccrs.PlateCarree())
    #    ax.set_yticks([0, 10, 20, 30, 40], crs=ccrs.PlateCarree())
    # Add scale bar
    scale_bar(ax=ax, length=500, location=(0.75, 0.075))
