from medphunc.image_analysis import image_utility as iu
from medphunc.image_analysis import clinical_mtf, segment_noise, clinical_nps
from medphunc.image_analysis import detectibility

from .eyeballer import is_lens_in_image, make_prediction

import pandas as pd
import pydicom
import scipy
import numpy as np
import os
import time
import imageio
from scipy.ndimage.morphology import binary_fill_holes
import json
import pathlib
import logging
logger = logging.getLogger(__name__)


def head_noise_wrapper(im: np.array, d: pydicom.Dataset) -> dict:
    nn = segment_noise.HeadNoise(im)
    results = {}
    results['noise'] = nn.results.loc['brain', 'noise']
    results['slice_thickness'] = d.SliceThickness
    return results


def mtf_wrapper(im: np.array, d: pydicom.Dataset) -> dict:
    
    mtf = clinical_mtf.clinical_mtf(im, [d.SliceThickness,*d.PixelSpacing])['Clinical']
    
    mtf_statistics = iu.calculate_resolution_from_MTF(mtf['frequency'], mtf['MTF'])
    results = {}
    results['mtf'] = mtf['MTF']
    results['mtf_frequency'] = mtf['frequency']
    results['mtf_50th_percentile'] = mtf_statistics['50\\%']
    results['mtf_10th_percentile'] = mtf_statistics['10\\%']
    return results


def test_angle_criteria(im: np.array, d: pydicom.Dataset) -> dict:
    """Measures the angle to vertical, but only if the other cosines are within limits
    Used to assess how well the patient has been angled to enable lens sparing
    """
    cosines = np.array(d.ImageOrientationPatient)
    angles = np.arccos(cosines)*180/np.pi
    
    results = {}
    
    results['angle_to_vertical'] = angles[4]
    
    #Should only use this angle if inverse cosine of the other
    #angles in this DICOM object are
    #(<10, 85-95, 85-95, 85-95, __, __)
    #and if 5th and 6th angles when added are <100
    #(should be pretty much 90°)
    
    conditions = []
    
    conditions.append(angles[0] < 10)
    conditions.append(all(angles[1:4] > 85) & all(angles[1:4] < 95))
    conditions.append(angles[4:].sum() < 100)
    
    results['angle_criteria_met'] = all(conditions)
    results['angle_condition1'] = conditions[0]
    results['angle_condition2'] = conditions[1]
    results['angle_condition3'] = conditions[2]
    
    return results
    
    
    

def assess_centrality(im: np.array, d: pydicom.Dataset) -> dict:
    image_position = np.array(d[('0020','0032')].value)
    yx_position = image_position[:-1][::-1]
    
    pixel_spacing = np.array(d.PixelSpacing).astype(np.float64)
    
    
    table_height = float(d.TableHeight)
    if d.Manufacturer == 'SIEMENS':
        yx_optimum = np.array([-table_height, 0])
    else:
        yx_optimum = np.array([0,0])
    
    yx_size = np.array(d.pixel_array.shape).astype(np.float64)
    
    yx_delta = yx_position + pixel_spacing*yx_size/2 - yx_optimum
    
    results = {}
    results['position_offset_y'] = yx_delta[0]
    results['position_offset_x'] = yx_delta[1]
    return results


def evaluate_lens_sparing(im: np.array, d: pydicom.Dataset) -> dict:

    lens_voxel_counts, sphere_cutoff = is_lens_in_image(im)
    results = {}
    results['lens_prediction'] = lens_voxel_counts
    results['lens_sphere_cutoff'] = sphere_cutoff
    results['lens_presence_confidence'] = make_prediction(lens_voxel_counts, sphere_cutoff)
    results['lens_presence_bool'] = results['lens_presence_confidence'] > 0.35
    
    return results




# def save_2d_image(im: np.array, d: pydicom.Dataset) -> dict:
#     # Save the image info stuff
    
#     save_fn = f'assessment_images/brain_{d.SOPInstanceUID}.png'
#     save_check_fn = f'assessment_images/brain_{d.SOPInstanceUID}_checkimage.png'
#     output = {}

#     try:
#         nn = segment_noise.HeadNoise(im)
#     except ValueError as e:
#         logger.debug('skipping %s because error during segmentation', str(e))
#         return output
    
#     z_export = locate_calcifications(im, nn.data['brain']['mask'])
    
#     if z_export > 0:
#         logger.debug(f'z level of {z_export} apepars to be ideal position. Exporting')
#         save_check_image(im, z_export, save_check_fn)
#         out = iu.apply_window(im[z_export,], 'brain',  unit_range=False)
#         imageio.imwrite(save_fn, out)
#         output['calcification_image_fn'] = save_fn
#         output['calcification_image_check_fn'] = save_check_fn
#     else:
#         logger.debug('skipping image because empty segmentation mask')
#         return output
#     return output

def clinical_detectibility_wrapper(im: np.array, d: pydicom.Dataset) -> dict:
    
    pixel_size = np.array(d.PixelSpacing)

    t = detectibility.TaskFunction(task_radius=1.5, task_contrast = 10, pixel_size = pixel_size[0])
    e = detectibility.EyeSol(t.f)
    ttf = clinical_mtf.clinical_mtf_from_dicom_metadata(im, d)
    nps = clinical_nps.automate_clinical_nnps(im, d)
    
    d = detectibility.Detectibility(nps, ttf, e, t)
    output = {}
    output['clinical_detectibility_index'] = d.d
    return output


tests = {#'save2dImage':save_2d_image,
        'clinicalMTF':mtf_wrapper,
        'HeadNoise':head_noise_wrapper,
        'AssessCentrality':assess_centrality,
        'testAngleCriteria':test_angle_criteria,
        'clinicalDetectibility':clinical_detectibility_wrapper
        }
    

def run_ctiq_functions(im, d):

    results = {}
    for test_name, test_function in tests.items():
        try:
            results[test_name] = test_function(im, d)
        except Exception as e:
            results[test_name] = {'error':str(e)}
    return results

        