from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import pickle

def sort_models(path,checksession,checkpoint):
    files = os.listdir(path)
    start_string = 'faster_rcnn_'+str(checksession)+'_'
    end_string = '_'+str(checkpoint)+'.pth'
    files = [file.replace(start_string,'').replace(end_string,'') for file in files if file.startswith(start_string)&file.endswith(end_string)]
    files = [int(x) for x in files]
    files.sort()
    #print(files)
    files = [str(x) for x in files]
    files = [start_string + file + end_string for file in files]
    print("Models correctly sorted")
    return files
