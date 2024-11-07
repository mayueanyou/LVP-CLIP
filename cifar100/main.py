import os,sys,torch,random,argparse
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Compose
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

current_path =  os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".")
upper_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
sys.path.append(upper_path)
from clipwrapper import ClipWrapper,ClipTransform


def validate_lvp(model_sel):
    clip = ClipWrapper(model_sel=model_sel)



if __name__ == '__main__':
    validate_lvp(7)