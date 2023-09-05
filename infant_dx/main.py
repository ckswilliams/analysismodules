import pydicom

from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu

from medphunc.image_analysis import image_utility as iu
import numpy as np

def calculate_low_variance_region_median_std(im):
    im = im.copy()
    large_window_std_array=iu.window_std(im,91)
    small_window_std_array=iu.window_std(im,9)
    
    background_region_threshold = threshold_otsu(im)

    null_mask = small_window_std_array==0
    background_region = im <background_region_threshold
    
    large_window_foreground_values = large_window_std_array[~background_region]
    variance_threshold = np.percentile(large_window_std_array[~background_region].flatten(),20) # cutoff at 20th percentile of foreground objects...

    low_variance_areas = (large_window_std_array<variance_threshold) & (~background_region)

    hopefully_liver_low_variance_mask = iu.find_largest_segmented_object(low_variance_areas)
    #plt.imshow(iu.draw_object_contours(draw=im, seg=hopefully_liver_low_variance_mask))
    median_pv = np.median(small_window_std_array[hopefully_liver_low_variance_mask])
    std = np.std(small_window_std_array[hopefully_liver_low_variance_mask])

    return {'low_variance_region_median_pv':median_pv, 'low_variance_region_std':std, 'detector_dose_metric':median_pv/std}


def infant_dx_iq(im, d):

    try:
        results  = calculate_low_variance_region_median_std(im)
    except Exception as e:
        results = {'error':str(e)}


    return results
