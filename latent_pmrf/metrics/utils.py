from pyiqa.default_model_configs import DEFAULT_CONFIGS
from .face_identity import FaceIdentity
from .landmark_distance import LandmarkDistance
from .fid_arch import FID
from .rmse_arch import MSE, RMSE

def register_metrics():
    DEFAULT_CONFIGS['mse'] = {
        'metric_opts': {
            'type': 'MSE',
        },
        'metric_mode': 'FR',
    }

    DEFAULT_CONFIGS['rmse'] = {
        'metric_opts': {
            'type': 'RMSE',
        },
        'metric_mode': 'FR',
    }

    DEFAULT_CONFIGS['face_identity'] = {
        'metric_opts': {
            'type': 'FaceIdentity',
        },
        'metric_mode': 'FR',
    }

    DEFAULT_CONFIGS['landmark_distance'] = {
        'metric_opts': {
            'type': 'LandmarkDistance',
        },
        'metric_mode': 'FR',
    }

    DEFAULT_CONFIGS['fid'] = {
        'metric_opts': {
            'type': 'FID',
        },
        'metric_mode': 'NR',
        'lower_better': True,
    }