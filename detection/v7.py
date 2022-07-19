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

def get_video_name(data):
    '''
    Extract video identfier from v7 data dict.

    Args: 
      data: dict holding v7 json annotations.
    '''
    return data['image']['original_filename'].replace('.mp4','').replace(' ','_')

def process_json(json_file, should_update_crop_spec=False, crop_w=FR_W, crop_h=FR_H, offset_x=1920-FR_W, offset_y=0):
    ''' Loads json, adds cropping specification should_crop is True.

    Args:
      json_file: str full path to json file.
      should_update_crop_spec: whether to write spec for "cropping_box" to json_file, which 
        the bounding box defining the cropped frame. Default spec is to crop to top right section of image.
      crop_w: width of cropping_box
      crop_h: height of cropping box
      offset_x: x coordinate for top left corner of cropping_box
      offset_y: y coordinate for top left corner of cropping_box
    Returns:
      dict holding json data
    '''

    with open(json_file, 'r') as f:
      data = json.load(f)

    print('Processing json for', get_video_name(data), '...')

    if not should_update_crop_spec:
        return data

    cropping_spec = data.get('cropping_box') # Returns None if not in data dict.
    if cropping_spec is None:
      print("Adding cropping_box field to json annotation...")
    else:
      print("Updating cropping_box field in json annotation...")

    image = data['image']
    data['cropping_box'] = {
        'width': crop_w,
        'height': crop_h,
        # x, y coordinates representing top left corner of cropped box, where top left corner of image is (0,0).
        'offset': {
            'y': offset_y,
            'x': offset_x
        }
    }
    with open(json_file, 'w') as f:
      json.dump(data, f, indent=2)
    return data

def write_positions(data, pos_dir=paths.POS_DIR, class_mapping={'dancing_bee':0}):
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
    '''
    video_name = get_video_name(data)
    make_dir(os.path.join(pos_dir, video_name))

    for video_annotation in data['annotations']:
        class_int = class_mapping.get(video_annotation['name']) # Returns None if not in class_mapping dict.
        if class_int is None:
          continue
        for frame, annotation in video_annotation['frames'].items():
            print('frame:', frame)
            a = annotation['directional_vector']['angle']
            a = ((math.degrees(a) + 90) + 360) % 360
            bb = annotation['bounding_box']
            h,w,x,y = bb['h'],bb['w'],bb['x'],bb['y']
            xc,yc = x+w/2,y+h/2
            cropping_spec = data.get('cropping_box') # Returns None if not in data dict.
            if cropping_spec:
                cropping_box = data['cropping_box']
                offset = cropping_box['offset']
                xc,yc = xc-offset['x'], yc-offset['y']
                # Do not write, if position is out of bounds of cropped frame.
                if xc < 0 or x >= cropping_box['width'] or yc < 0 or y >= cropping_box['height']:
                    continue
            with open(os.path.join(pos_dir, video_name, "%06d.txt" % int(frame)), 'a') as f:
                np.savetxt(f, [[xc,yc,class_int,a]], fmt='%i', delimiter=',', newline='\n')

######## MAIN FUNCTIONS ##############

def import_annotations(v7_annotations_file=None, v7_annotations_dir=None, pos_dir=paths.POS_DIR, should_update_crop_spec=False, crop_w=FR_W, crop_h=FR_H, offset_x=1920-FR_W, offset_y=0, class_mapping={'dancing_bee':0}):
    '''
    Process a v7 json file(s) and write position .txt files in format that dataset.py accepts.

    Args:
      v7_annotations_dir: dir holding json files.
      pos_dir: output dir with sub directories holding .txt files. sub directory name is taken from filename stored in json.
      Rest of args is documented in process_json() and write_positions() functions above.
    '''
    if (v7_annotations_file and v7_annotations_dir) or not (v7_annotations_file or v7_annotations_dir):
        print("Warning: need to specify either `v7_annotations_file` or `v7_annotations_dir`")
        return
    json_files = [v7_annotations_file]
    if v7_annotations_dir:
      json_files = func.get_all_files([v7_annotations_dir])
    for json_file in json_files:
      data = process_json(json_file, should_update_crop_spec, crop_w, crop_h, offset_x, offset_y)
      write_positions(data, pos_dir, class_mapping)

def create_frames_from_video(video_path, img_dir=paths.IMG_DIR, v7_annotations_dir=None, frame_range=None):
    '''
    Write frame .png files to sub dir under img_dir.

    Crops frames according to cropping spec in json file stored under v7_annotations_dir, if provided.
    

    Args:
      video_path: full path to .mp4 file
      img_dir: output dir to store frames, frames are stored in a sub dir of img_dir called video_name
      v7_annotations_dir: optional input dir holding v7 json annotation files, with cropping specs.
          Expects v7_annotations_dir to hold json files the same name stem as the video filename.
      frame_range: range of sequential frames numbers (indexed by 0), if empty produces frames for entire video
    '''
    video_name = video_path.split('/')[-1].replace('.mp4','')
    print('Processing video', video_name, '...')
    make_dir(os.path.join(img_dir,video_name))
    cap = cv2.VideoCapture(video_path)
    print("total frames",cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cropping_spec = None
    if v7_annotations_dir is not None:
        json_file = os.path.join(v7_annotations_dir, video_name + '.json')
        with open(json_file, 'r') as f:
            annotation_data = json.load(f)
        cropping_spec = annotation_data.get('cropping_box') # Returns None if not in data dict.

    if frame_range is None:
      frame_range = range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i, frame_i in enumerate(frame_range):
        print('frame:',frame_i)
        frame = func.get_frame_from_video_capture(frame_i, cap)
        if i==0: print('frames shape:',frame.shape)
        # Crop frame.
        if cropping_spec is not None:
            crop_h, crop_w, offset = cropping_spec['height'],cropping_spec['width'],cropping_spec['offset']
            xo,yo=offset['x'],offset['y']
            frame = frame[yo:yo+crop_h, xo:xo+crop_w]
            if i==0: print('cropped frames shape:',frame.shape)
        # Save frame.
        im = Image.fromarray(frame)
        im.save(os.path.join(img_dir, video_name,"%06d.png" % frame_i))

def create_frames(video_dir, img_dir=paths.IMG_DIR, v7_annotations_dir=None, frame_range=None):
    '''
    Write frame .png files to sub dirs under img_dir, one sub dir per video.

    Crops frames according to cropping spec in json file stored under v7_annotations_dir, if provided.

    Args:
      video_dir: input dir holding .mp4 files
      img_dir: output dir to store frames, frames are stored in a sub dir of img_dir called video_name
      v7_annotations_dir: optional input dir holding v7 json annotation files, with cropping specs.
          Expects v7_annotations_dir to hold json files the same name stem as the video filename.
      frame_range: range of sequential frames numbers (indexed by 0), if empty produces frames for entire video
    '''
    video_files = func.get_all_files([video_dir])
    for video_file in video_files:
        create_frames_from_video(video_file, img_dir, v7_annotations_dir, frame_range)