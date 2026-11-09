from multiprocessing.dummy import Pool as threadpool
import cv2 
import torch .nn as nn 
import torch.nn.functional as F 
import torchvision
import torch.utils.data as data 
import os 
import json
import tqdm.auto as tqdm 
import time 
import matplotlib.pyplot as plt 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
import slideio
import progressbar
from pathlib import Path
import imageio
import shapely.geometry as sg
import argparse
import pandas as pd
from math import sqrt 
try:
    import openslide as ops
except ModuleNotFoundError:
    ops = None



#### Data-Extraction annotation pipeline, Converting Slide Images to Tiles .

Image.MAX_IMAGE_PIXELS = 100000000000
DEFAULT_JPG_MPP = 0.245
NUM_THEADS = 8
JSON_ANNOTATION_SCALE = 15 


class Annotationoject:
    def __init__(self, name):
        self.name = name 
        self.coordinate = [] ### it contain annotated dataset which are x, y respectively ####
        
    def add_cord(self, cord):
        return self.coordinate.append(cord) #### it contains cord per each value of the pathological dataset ####
    
    
    def add_scale(self, scale):
        return np.multiply(self.coordinate, 1/scale)### values for rescale the input values(x), and output values(y)
    
    def print_coord(self):
        for p in self.coordinate:
            print(p)   #### the number of point in each cord 
            
            
    def add_shape(self, shape):
        for point in shape:
            self.add_cord(point) #### adding shape to the dimension of the pathology 
        
        
class JPG_Slider:
    def __init__(self, path, mpp):
      self.load_image = imageio.imread(path)
      self.dimension = (
            [self.load_image.shape[1],self.load_image.shape[0]]
        )
        
      self.values = {"openslide.mpp-x": mpp}
      self.dimensions = [self.dimension]
      self.count_values = 1
    
    def get_thumbvalues(self, dimension):
        return cv2.resize(self.load_image, dsize=self.dimension, interpolation = cv2.INTER_CUBIC)
    
    def get_values(self, topleft, level, window):
        return self.load_image[topleft[1]:topleft[1] + window[1],
                                    topleft[0]:topleft[0] + window[0],]
        
        
        """
        Args: 
        Reading JPG_Slider images, that involves loading the images
        using self_load_images.
        self.dimension_image :
        get the shape of each loaded images 
        self.count_values :
        get the count of each values 
        
        get_thumbvalues:
        return the dimension of each thumb 
        using cv inbuilt library .
        
        get_values :
        get the values frome each loaded jpg images
        """
        
        
class Slider_reader:
    def __init__(self, path, filetype, export_folder = None, pp =   None):
        self.coord = []
        self.filetype = filetype
        self.export_folder = export_folder
        self.pp = pp
        self.p_id = []
        self.extract_px = None
        self.annsPoly = []
        self.shape = None
        self.basename = os.path.splitext(path)[0]
        self.name = os.path.basename(self.basename)
        self.has_annotation = True
        self.ignorefile = []
        self.notabletoload = False
        
        
        if filetype in ["svs", "tif", "ndpi", "vms", "vmu", "mrxs"]:
            try:
                if ops is None:
                    raise ModuleNotFoundError(
                        "OpenSlide is required for whole-slide formats. "
                        "Install the openslide-bin package on Windows."
                    )
                self.slide = ops.OpenSlide(path)
            except Exception as exc:
                print("unable to load slide " + filetype + " " + path + ": " + str(exc))
                self.notabletoload = True 
                return None
                
                
        if filetype == "jpg":
            try:
                self.slide = JPG_Slider(path, mpp=DEFAULT_JPG_MPP)
            except Exception as exc:
                print("unable to load slide " + filetype + " " + path + ": " + str(exc))
                self.notabletoload = True
                return None
            
            thumb_file = os.path.join(export_folder or os.path.dirname(path), "thumb")
            if not os.path.exists(thumb_file):
                os.makedirs(thumb_file, exist_ok = True)
             
             
            ##### get roi files for each slide image (pdf and csv files) ####
            
            ros_file_csv = self.basename + ".csv"
            ros_file_pdf = self.basename + ".pdf"
            
            
            if os.path.exists(ros_file_csv) and os.path.getsize(ros_file_csv) > 0:
                self.has_annotation = True
                self.annotation_file = ros_file_csv 
                self.annotation_type = "csv"
                
                
            if os.path.exists(ros_file_pdf) and os.path.getsize(ros_file_pdf) > 0:
                self.has_annotation = True 
                self.annotation_type = ros_file_pdf
                self.annotation_type = "pdf"
                
                
                
                """
                Args : slider_reader files contain 
                annotation_polygon: which contain my annotation polygon files for the annnotated files 
                basename: which contain the path of the file folder in which the root of the files are located
                shape: which contain the shape (H, W, D) of whole_slide imaging file
                extract_px: extracting pixels from each files 
                filetype: which contain different types of files[svs,vmu, mrxs,vms,jpg]
                ros_file_csv : it is located in the basefilename 
                ros_file_pdf: it is located in the basefilename  
                """

        if self.notabletoload:
            return
        if filetype == "jpg":
            self.shape = tuple(self.slide.dimension)
            self.MPP = DEFAULT_JPG_MPP
        else:
            self.shape = tuple(self.slide.dimensions)
            mpp = self.slide.properties.get("openslide.mpp-x")
            self.MPP = float(mpp) if mpp else DEFAULT_JPG_MPP
                
                
                
                
    def load_file(self):
        return bool(self.shape)
    
    
    def build_generator(self, size_px, size_um, stride_div, case_name, tile_path, category, export = False, augment = False):
        if self.MPP <= 0:
            raise ValueError("MPP must be greater than zero")
        if stride_div <= 0:
            raise ValueError("stride_div must be greater than zero")
        self.extract_px = int(size_um / self.MPP)
        stride = int(self.extract_px * stride_div)
        
        
        slide_x_size = self.shape[0] - self.extract_px
        slide_y_size = self.shape[1] - self.extract_px
        
        
        print("slide_x_size", slide_x_size)
        print("slide_y_size", slide_y_size)
        
        for y in range(0, (self.shape[1]+1) - self.extract_px, stride):
            for x in range(0, (self.shape[0] + 1) - self.extract_px, stride):
                self.coord.append((x, y))
                
                
    """
    Args:
    slide_x_size : generating slide from x-values(input_values)
    slide_y_size: generating slide from the y_values(target-values)
    self.extract_px: extracting pixels values from x_values
    """
    
    
def get_generator(self):
    for ci in range(len(self.coord)):
        c = self.coord[ci]
    print("c:values", c)
    
    
    
    
def ros_load_csv(self, path):
    with open(path, "r") as json_file:
        json_data = json.load(json_file)["shapes"]
    for shape in json_data:
        area_reduced = np.multiply(shape["points"], JSON_ANNOTATION_SCALE)
    print("area_reduced", area_reduced)
    
    
    
def ros_load_pdf(self, path):
    reader = pd.read_csv(path)
    headers = []
    for col in reader.columns:
        headers.append(col.strip())
    if "X_base" in headers and "Y_base" in headers:
        index_x = headers.index("X_base")
        index_y = headers.index("Y_base")
    else:
        raise ValueError("CSV must contain X_base and Y_base columns")
        
    print("index_x", index_x)
    print("index_y", index_y)
    
    
    
    

class Convoluter:
    def __init__(self, size_px, size_um, stride_div, save_folder = "", skipws = False, augment = False):
        
        self.Slides = []
        self.SLIDES = {}
        self.size_px = size_px
        self.stride_dv = stride_div 
        self.save_folder = save_folder 
        self.size_um = size_um 
        self.augment = augment 
        self.skipws = skipws
        
        
    def load_slides(self, slide_array, directory = "None", category = "None"):
        self.filesize = len(slide_array)
        self.iterator = 0
        print(f"the total number of slide in the folder:" + str(self.filesize))
        
        
        
        for slide in slide_array:
            name = slide.split(".")[:-1]
            name = ".".join(name)
            name = name.split(".")[-1]
            filetype = slide.split(".")[-1]
            
            path = slide 
            
            self.SLIDES.update({name:{
                "name": name,
                "path":path,
                "type":filetype,
                "category":category
            }}) 
            
        return self.SLIDES 
    
    
    
    
    
    
    
def get_args():
 parser = argparse.ArgumentParser(description = "the script for generating whole-slide imagin (WSI)")
 parser.add_argument(
    "-s", "--slide", help = "path to whole-slide-imaging (SVS or JPG FORMAT) or folder to generate the images (SVS or JPG) to analyze"
)
 parser.add_argument("-o", "--out", help = "path to the directory in which the whole-slide-imaging are being generated")
 
 parser.add_argument("-ov", type = float, default = 1.0, help = "the size of overlapping can be between 1 and 0")
 
 parser.add_argument("--augment", action = "store_true", help = "applying different values of augementation in transforming the images it includes extracted tiles with flipping/rotations")
 
 
 if __name__ == ("__main__"):
     args = get_args()
     print("args", args)
        
        
    
                
                
                 
        
        
        
        
        
        
        
        
                
                
                
            

            
        
        
        
        
        
        
        
    
        
        
        
        
        
        
