from fastapi import FastAPI
from typing import Optional, List

from pydantic import BaseModel
import numpy as np
import pydicom

import json

from medphunc.pacs import thanks
from medphunc.image_io import ct

from ct_brain_iq.main import run_ctiq_functions, evaluate_lens_sparing

import logging
# setup loggers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)  # the __name__ resolve to "main" since we are at the root of the project. 
                                      # This will get the root logger since no logger in the configuration has this name.


#thanks.pi.MY.set_from_saved('IMGTLBX01')



class Task(BaseModel):
    study_instance_uid_list: List[str]
    series_instance_uid_list: List[str]
    task_name: str
    accession_number: Optional[str] = None
    series_name: Optional[str] = None



class Result(BaseModel):
    study_instance_uid: str
    series_instance_uid: str
    task_name: str
    status: str # Status should be returned as: processing or complete
    error: Optional[str] = None
    json_data: Optional[str] = None
    


from medphunc.pacs import thanks


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/")
async def run_task(task: Task):
    logger.info('Task received')
    logger.debug(task)
    result_data = run_task(task)

    result = Result(study_instance_uid = task.study_instance_uid_list[0],
                    series_instance_uid = task.series_instance_uid_list[0],
                    status = 'complete',
                    task_name = task.task_name,
                    json_data = json.dumps(result_data, cls=CustomJSONizer))
    return result
    
    
class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        elif isinstance(obj, np.ndarray):
            return super().encode(obj.tolist())
        else:
            return super().default(obj)
    
    
def run_task(task) -> bool:
    """
    sadss

    """
    logger.debug('Running task')
    if task is None:
        return True
    if task.task_name not in tests.keys():
        raise(NotImplementedError('No ' +task.task_name + ' available'))
    fnc = tests[task.task_name]
    print('got fnc')
    try:
        im, d, meta = thanks.retrieve_series(task.series_instance_uid_list[0])
    except ValueError:
        tt = thanks.Thank('series',StudyInstanceUID = task.study_instance_uid, SeriesInstanceUID=task.series_instance_uid_list[0])
        tt.move()
        im, d, meta = thanks.retrieve_series(task.series_instance_uid_list[0])
    except Exception as e:
        return {'error': str(e)}
        
    try:
        #logger.debug('Analysing series key %s Running task %s', task)
        result = fnc(im=im, d=d)
    except Exception as e:
        logger.error('Task failed' + str(e))
        return {'error': str(e)}
    logger.debug('Completed task')
    return result



tests = {'CTBrainQuality':run_ctiq_functions,
         'EyeLens':evaluate_lens_sparing}