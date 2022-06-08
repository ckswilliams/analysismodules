# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:07:27 2020

@author: willcx
"""


import numpy as np
from medphunc.image_io import ct
from medphunc.image_analysis import image_utility as iu
from medphunc.image_analysis import segmentation
import glob
import pandas as pd

from scipy.ndimage import binary_fill_holes, morphology, binary_opening, binary_closing, binary_erosion, binary_dilation, generate_binary_structure

from skimage import measure
import scipy

#%% debug
import imageio
from matplotlib import pyplot as plt


#%% settings

#y 0 134/596 (.225)
Y_RANGE = np.array([0, .33])

#z 0:half
Z_RANGE = np.array([0, 0.5])

# eyeball HU range
THRESH_RANGE = np.array([-10,15])

#lens HU min
LENS_RANGE = np.array([50, 90])

#%%








#%% Draw a sphere roughly encompassing the eyeball

def spherey(shape, center, radii):
    "define a sphere. shape is y,x,z size. Center/radii are array of y,x,z values"
    X = np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')
    X = np.moveaxis(np.array(X),0,3)
    return (((X - center) / radii)**2).sum(axis=3) <= 1



#%%

import skimage

def segment_eyeballs(roi):
    
    processed_roi = iu.apply_window(roi, 'brain', unit_range=False)
    
    
    k = spherey((3,5,5), (1,2,2), (1,2,2))
    processed_roi = scipy.ndimage.filters.median_filter(processed_roi, footprint=k)
    processed_roi = scipy.ndimage.filters.median_filter(processed_roi, footprint=k)
    #roi = scipy.ndimage.filters.median_filter(roi,size=3)

    seg = (roi > THRESH_RANGE[0]) & (roi < THRESH_RANGE[1])
    seg = skimage.morphology.remove_small_objects(seg)
    
    
    k = generate_binary_structure(2,1)[np.newaxis,:,:]
    k = spherey((1,5,5), (1,2,2), (2,3,3))
    seg = binary_opening(seg, iterations = 2, structure=k)
    seg = binary_closing(seg, iterations = 2, structure=k)
    #seg = binary_opening(seg, iterations = 2, structure=k)
    #seg = binary_closing(seg, iterations = 2, structure=k)

    k = spherey((3,5,5), (1,2,2), (1,2,2))
    
    seg[:,:,seg.shape[2]//2] = False
    
    #seg = binary_opening(seg, iterations = 1, structure=k)
    #seg = binary_opening(seg, iterations = 1, structure=k)
    # seg = binary_closing(seg, iterations = 2, structure = k)
       
    #k = generate_binary_structure(2,1)[np.newaxis,:,:]

    #seg = binary_closing(seg, iterations = 4, structure=k)
    #seg = binary_opening(seg, iterations = 4, structure=k)
    
    #segmentation.show_all_slices(seg)
    
    # look for paired objects in x direction
    sd = seg.shape[2]//11
    #create a weighting function that has 2 peaks around the expected location of the eyes
    x_weight = scipy.signal.windows.cosine(seg.shape[2]) - scipy.signal.windows.gaussian(seg.shape[2],sd)
    x_weight = x_weight**0.5
    x_weight = x_weight - .2
    
    weighted_seg = seg * x_weight[None,None,:]
    
    # Create a weighting function in z. peaks at the z which has the highest sum after applying the x weighting
    # used to try to make sure we get the eyes as a pair, and not 1 eye and some other object when there is cutoff
    z_max_seg = weighted_seg.sum(axis=1).sum(axis=1).argmax()
    
    z_weight = z_max_seg - np.abs(z_max_seg - np.arange(seg.shape[0]))
    z_weight = z_weight-z_weight.min()
    z_weight = z_weight/z_weight.max()
    z_weight = z_weight ** 2
    weighted_seg = weighted_seg * z_weight[:,None,None]

    # Finally, weight in y. Assume that the head is about 2/3 of max thickness in the x direction at the best y location
    y_sums = (processed_roi[z_max_seg,]>processed_roi.min()).sum(axis=1)
    y_sums = y_sums > (y_sums.max()*.7)
    
    best_y_index = y_sums.argmax()
        
    y_weight = best_y_index - np.abs(best_y_index - np.arange(seg.shape[1]))
    y_weight[y_weight < 0] = 0
    y_weight = y_weight ** 0.4
    
    weighted_seg = weighted_seg * y_weight[None,:,None]
    
    # Look at all the obects in the segmented image
    t, n = measure.label(seg, return_num=True)
    # measure the properties using the weighted segmentation as the intensity image
    props = measure.regionprops(label_image=t, intensity_image=weighted_seg)
    
    # Choose eyeballs based on the highest total intensity after applying the weighting factors above
    weighted_areas = [pp.mean_intensity * pp.area for pp in props]
    area_indices_sorted = np.argsort(weighted_areas)
    eyeball_props = (props[area_indices_sorted[-1]], props[area_indices_sorted[-2]])
    
    eyeball_masks = [t==prop.label for prop in eyeball_props]
    
    return eyeball_masks


def spherey(shape, center, radii):
    "define a sphere. shape is y,x,z size. Center/radii are array of y,x,z values"
    X = np.meshgrid(*[np.arange(i) for i in shape], indexing='ij')
    X = np.moveaxis(np.array(X),0,3)
    return (((X - center) / radii)**2).sum(axis=3) <= 1


def make_eyeball_spheremask(eyeball_mask):
    eyeball_mask = eyeball_mask.copy()

    p = measure.regionprops(eyeball_mask.astype(int))[0]
    try:
        p_convex = p.convex_image
    except:
        p_convex = p.image
            
    # s_data = iu.localise_segmented_object(eyeball_mask)

    # Z1 = np.where(eyeball_mask[:,:,s_data['x_range'][0]])
    # Z2 = np.where(eyeball_mask[:,:,s_data['x_range'][1]])
    # Z3 = np.where(eyeball_mask[:,s_data['y_range'][0],:])
    # Z4 = np.where(eyeball_mask[:,s_data['y_range'][1],:])
    
    # z_center = np.mean(np.concatenate([Z1[0], Z2[0], Z3[0], Z4[0]]))
    # x_center = np.mean(np.concatenate([Z3[1], Z4[1]]))
    # y_center = np.mean(np.concatenate([Z1[1], Z2[1]]))
    
    # Since the eyeball might be cut off, we can't take the centroid to get the z center
    # Find the widest part of the mask as the central z, using the convex image
    z_center = p_convex.sum(axis=1).sum(axis=1).argmax() + p.bbox[0]
    
    # take half the major axis length as the radius
    radius = p.major_axis_length / 2
    
    # method 1: distance thickest part to to top
    z_rad_1 = p.image.shape[0]+p.bbox[0]-z_center
    
    #method 2: half the region size in z
    z_rad_2 = p.image.shape[0]/2
    
    #Take the max of the two methods or 2, because less than 2 results in no mask
    z_rad = max(z_rad_1, z_rad_2, 2)
    
    center = np.array([z_center, *p.centroid[1:]])
    #center = [y_center, x_center, z_center]
    
    #make the eyeball sphere mask
    eye_overmask = spherey(eyeball_mask.shape, center, np.array([z_rad,radius,radius]))
    
    #Open it a little, just in case
    k = generate_binary_structure(3,1)
    eye_overmask = binary_dilation(eye_overmask, k, iterations = 1)
    
    return eye_overmask


def calculate_lens_presence_metric(im, eyeball_spheremask):
        
    lens = (im > LENS_RANGE[0]) & (im < LENS_RANGE[1]) & eyeball_spheremask
    
    vals = im[lens]
    return vals.shape[0]

#%%
def is_lens_in_image(im):
    "Look for eye lens around top of CT image, count number of qualifying pixels"
    #Restrict image to the approximate ROI
    head_pos = iu.localise_phantom(im)
    
    y_range_i = np.array((head_pos['y_range'][0], np.diff(head_pos['y_range'])[0]*Y_RANGE[1] + head_pos['y_range'][0])).astype(np.int32)
    z_range_i = (im.shape[0] * Z_RANGE).astype(np.int32)
    
    roi = im[z_range_i[0]:z_range_i[1],y_range_i[0]:y_range_i[1],:]
    
    eyeball_masks = segment_eyeballs(roi)
    
    # Code for 'was it actually an eye we found?'

    p1 = measure.regionprops(eyeball_masks[0].astype(int), roi)[0]
    p2 = measure.regionprops(eyeball_masks[1].astype(int), roi)[0]

    if any(m.sum() < 2000 for m in eyeball_masks):
        # If the eyeball masks are too small, assume they are deficient and 
        # Give passing grade to case
        print('assuming deficient eyeball masks?')
        #plt.imshow(iu.apply_window(p1.image[0,], 'brain'))
        #plt.show()
        return [[0,0],[1,1]]
    

    #Reorder eyeballs to left, right
    order_vals = [m.argmax(axis=2).max() for m in eyeball_masks]
    if order_vals[0] > order_vals[1]:
        eyeball_masks = (eyeball_masks[1], eyeball_masks[0])
    
    spheremasks = [make_eyeball_spheremask(m) for m in eyeball_masks]
    
    #method 1: look for HU values around 50-90 in the spheremask
    
    lens_values = [calculate_lens_presence_metric(roi, sm) for sm in spheremasks]
    
    lens_prediction = np.array(lens_values)
    
    #method 2:look to see if any part of the sphere is in the cut off. if no, 
    #lens must be in the beam
    sphere_cutoff = [(roi[m] == -1024).sum() / m.sum() for m in spheremasks]
    
    return lens_prediction, sphere_cutoff


def make_prediction(lens_voxel_counts, sphere_cutoff):
    "Predict whether the eye lens is in the image. Higher number means more sparing. 0 means both lens probably spared. 1 means no sparing"
    # if sphere cutoff is 0.4, assume full sparing
    # If 0, no sparing
    # linear between those values
    x1, x2 = (0, 0.25)
    sphere_prediction = (np.array(sphere_cutoff) - x1) / (x2-x1)
    sphere_prediction[sphere_prediction>1] = 1
    sphere_prediction[sphere_prediction<0] = 0
    sphere_prediction = sphere_prediction.sum()

    # lens_voxel_sum = np.array(lens_voxel_counts).sum()
    # # If lens voxels identified, increase by up to 0.5.
    # # If no lens voxels, or less than 400, decrease by up to 0.25
    # lens_voxel_prediction = lens_voxel_sum/400 - 0.25
    # lens_voxel_prediction = max(min(lens_voxel_prediction,0.5),0)
    
    # prediction_value = sphere_prediction + lens_voxel_prediction
    # prediction_value = min(prediction_value, 1)
    # prediction_value = max(prediction_value, 0)
    prediction_value = (2 - sphere_prediction)/2
    
    return prediction_value

#%%

