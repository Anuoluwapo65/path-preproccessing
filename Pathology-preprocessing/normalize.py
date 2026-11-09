import torch 
import os  
import multiprocessing
from multiprocessing import Pool as threadpool
import imageio
import cv2
import progressbar
from pathlib import Path 
import matplotlib
from math import sqrt 
import sys
import glob
import argparse
from itertools import repeat
from contextlib import contextmanager



def normalize_Main(inputPath, outputPath, item, normalizer):
    
    outputPathRoot = os.path.join(outputPath, item)
    inputPathRoot = os.path.join(inputPath, item)
    
    if(os.path.isdir(inputPathRoot)):
        inputPathRootContent = os.listdir(inputPathRoot)
        if not len(inputPathRootContent) == 0:
            if not os.path.exists(outputPathRoot):
                os.makedirs(outputPathRoot)
                
            temp = os.path.join(inputPath, item)
            tempcontent = os.listdir(temp)
            
            print("tempcontent", tempcontent)
        
        
        
        
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()
    
    
              
