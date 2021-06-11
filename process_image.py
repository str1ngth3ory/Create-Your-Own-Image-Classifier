import numpy as np
import torch
from PIL import Image

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image_path)

    # Resize to 256 while keeping aspect ratio
    box = img.getbbox()[-2:]
    if box[0] <= box[1]:
        box = (256, int(256/box[0]*box[1]))
    else:
        box = (int(256/box[1]*box[0]), 256)
    img = img.resize(box)
    
    # Crop 224x224 out of the center
    img = img.crop(((box[0]-224)/2, (box[1]-224)/2, (box[0]+224)/2, (box[1]+224)/2))
    
    # Convert to numpy array and normalize
    np_img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img/255-mean)/std
    np_img = np_img.transpose(2,0,1)
    
    return torch.from_numpy(np_img).float() # Convert to FloatTensor