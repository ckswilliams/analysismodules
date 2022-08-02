

from medphunc.image_analysis import water_equivalent_diameter
import numpy as np
import pydicom


def wed(im: np.array, d: pydicom.Dataset) -> dict:
    
    result = water_equivalent_diameter.wed_from_image(im, [*d.PixelScale, d.SliceThickness])
    
    return {'water_equiv_circle_diam':result['water_equiv_circle_diam']}