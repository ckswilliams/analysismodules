

from medphunc.image_analysis.water_equivalent_diameter import WED
import numpy as np
import pydicom


def wed(im: np.array, d: pydicom.Dataset) -> dict:
    
    wed = WED.from_volume(im, d, method='full')
    wed.wed_results['ssde'] = wed.ssde.iloc[0]
    
    return wed.wed_results


