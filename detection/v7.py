''' Library to import annotations from V7 software and image frames from video files into a format that dataset.py expects.
'''
import cv2
import json
import math
import numpy as np
import os
from PIL import Image
from utils.func import FR_H, FR_W, make_dir
from utils import func
from utils import paths

class CroppingSpec:
    def __init__(self, width, height, top_left_x, top_left_y):
      self.width = width     # width of cropping box
      self.height = height   # height of cropping box
      self.offset_x = top_left_x    # x coordinate for top left corner of cropping box (top left is origin)
      self.offset_y = top_left_y    # y coordinate for top left corner of cropping box (top left is origin)

def get_video_name(data):
    '''
    Extract video identfier from v7 data dict.

    Args: 
      data: dict holding v7 json annotations.
    '''
    return data['image']['original_filename'].replace('.mp4','').replace(' ','_')

def read_json(json_file):
    ''' Loads json, adds cropping specification should_crop is True.

    Args:
      json_file: str full path to json file.
    Returns:
      dict holding json data
    '''

    with open(json_file, 'r') as f:
      data = json.load(f)

    print('Processing json for', get_video_name(data), '...')

    return data

def write_positions(data, pos_dir=paths.POS_DIR, class_mapping={'dancing_bee':0}, cropping_spec=CroppingSpec(width=FR_W, height=FR_H, top_left_x=1920-FR_W, top_left_y=0)):
    '''Compute and write position coordinates & angle from v7 data dict.
    
    Expects V7 annotations to be a polygon or bounding box with directional vector.
    Grabs center of bounding box and offsets if cropping spec is specified.
    Grabs directional vector and converts from [angle from horizontal] to [angle from vertical].
    Writes one .txt file per frame, where each file holds all bee annotations for that frame and each row
    is one bee annotation holding: x center, y center, bee class (0), angle (deg) clockise from the vertical.

    Args:
      data: dict holding annotation data in v7 format.
      pos_dir: str full path of the output directory.
      class_mapping: dict mapping str name of bee class in v7 tool to an integer encoding.
      cropping_spec: cropping specification (of type CroppingSpec), used to offset the bee center coordinates. Expects None if no cropping is applied.
    '''
    video_name = get_video_name(data)
    make_dir(os.path.join(pos_dir, video_name))

    for video_annotation in data['annotations']:
        class_int = class_mapping.get(video_annotation['name']) # Returns None if not in class_mapping dict.
        if class_int is None:
          continue
        for frame, annotation in video_annotation['frames'].items():
            print('frame:', frame, end='...')
            a = annotation['directional_vector']['angle']
            a = ((math.degrees(a) + 90) + 360) % 360
            bb = annotation['bounding_box']
            h,w,x,y = bb['h'],bb['w'],bb['x'],bb['y']
            xc,yc = x+w/2,y+h/2
            # If cropping specified, offset bee center.
            if cropping_spec is not None:
                xc,yc = xc - cropping_spec.offset_x, yc - cropping_spec.offset_y
                # Do not write, if position is out of bounds of cropped frame.
                if xc < 0 or xc >= cropping_spec.width or yc < 0 or yc >= cropping_spec.height:
                    print("Offset bee center (%d,%d) outside cropping box (%d, %d)" % (xc,yc, cropping_spec.width, cropping_spec.height))
                    continue
            with open(os.path.join(pos_dir, video_name, "%06d.txt" % int(frame)), 'a') as f:
                np.savetxt(f, [[xc,yc,class_int,a]], fmt='%i', delimiter=',', newline='\n')

######## MAIN FUNCTIONS ##############

def import_annotations(v7_annotations_file, pos_dir=paths.POS_DIR, class_mapping={'dancing_bee':0}, cropping_spec=None):
    '''
    Process a v7 json file and write position .txt files in format that dataset.py accepts.

    Args:
      v7_annotations_file: json file from v7 application.
      pos_dir: output dir with sub directories holding .txt files. sub directory name is taken from filename stored in json.
      cropping_spec: cropping specification (of type CroppingSpec), used to offset the bee center coordinates. Expects None if no cropping to the video frame will applied.
    '''

    data = read_json(v7_annotations_file)
    write_positions(data, pos_dir, class_mapping, cropping_spec)

def create_frames_from_video(video_path, img_dir=paths.IMG_DIR, frame_range=None, cropping_spec=None):
    '''
    For one video, write frame .png files to subdir under img_dir.

    Crops frames according to cropping spec, if provided.
    

    Args:
      video_path: full path to .mp4 file
      img_dir: output dir to store frames, frames are stored in a sub dir of img_dir called video_name
      frame_range: range of sequential frames numbers (indexed by 0), if empty produces frames for entire video
      cropping_spec: cropping specification (of type CroppingSpec), used to crop the frame. Expects None if no cropping is applied.
    '''
    video_name = video_path.split('/')[-1].replace('.mp4','')
    print('Processing video', video_name, '...')
    make_dir(os.path.join(img_dir,video_name))
    cap = cv2.VideoCapture(video_path)
    print("total frames",cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_range is None:
      frame_range = range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i, frame_i in enumerate(frame_range):
        print('frame:', frame_i, end='...')
        frame = func.get_frame_from_video_capture(frame_i, cap)
        if i==0: print('frames shape:',frame.shape)
        # Crop frame.
        if cropping_spec is not None:
            crop_h, crop_w = cropping_spec.height,cropping_spec.width
            xo, yo=cropping_spec.offset_x, cropping_spec.offset_y
            frame = frame[yo:yo+crop_h, xo:xo+crop_w]
            if i==0: print('cropped frames shape:',frame.shape)
        # Save frame.
        im = Image.fromarray(frame)
        im.save(os.path.join(img_dir, video_name,"%06d.png" % frame_i))