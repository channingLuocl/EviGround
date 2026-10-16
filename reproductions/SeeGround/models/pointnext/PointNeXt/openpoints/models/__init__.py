"""
Author: PointNeXt

"""
# from .backbone import PointNextEncoder
from .backbone import *
from .segmentation import * 
from .classification import BaseCls
import os
# Classification inference does not need the reconstruction-only Chamfer CUDA extension.
if os.environ.get('SEEGROUND_POINTNEXT_INFERENCE_ONLY') != '1':
    from .reconstruction import MaskedPointViT
from .build import build_model_from_cfg
