''' Library to import annotations from V7 software and image frames from video files into a format that dataset.py expects.
'''
import cv2
import json
import math
import numpy as np
import os
from PIL import Image
from utils.func import FR_D, FR_H, FR_W, make_dir
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

def get_video_filename(data):
    '''
    Extract video file name from v7 data dict.

    Args:
      data: dict holding v7 json annotations.
    '''
    return data['image']['original_filename'].replace(' ','_')

def find_cropping_spec(v7_annotations_file, crop_w=FR_D, crop_h=FR_D):
  '''
  Given annotations of a video and the crop dimensions, find appropriate cropping offsets based on the bee's movement.

  v7_annotations_file: full path of json file from v7 application.
  crop_w: width of resulting cropped frame
  crop_h: height of resulting cropped frame
  '''
  bbx0, bbx1, bby0, bby1 = [], [], [], []
  with open(v7_annotations_file) as f:
    # load each video annotations
    json_object = json.load(f)
    frame_width, frame_height = json_object['image']['width'], json_object['image']['height']
    annotations = json_object['annotations']
    # load each group of annotations
    for group in annotations:
      # load each frame info
      for frame_id in group['frames']:
        frame = group['frames'][frame_id]
        # bb corners
        bbx0.append(frame['bounding_box']['x'])
        bbx1.append(frame['bounding_box']['x']+frame['bounding_box']['w']-1)
        bby0.append(frame['bounding_box']['y'])
        bby1.append(frame['bounding_box']['y']+frame['bounding_box']['h']-1)

  # get max extent
  min_x, max_x = int(min(bbx0)), round(max(bbx1))
  min_y, max_y = int(min(bby0)), round(max(bby1))
  # compute what to add
  toAdd_x = crop_w - (max_x-min_x+1)
  toAdd_y = crop_h - (max_y-min_y+1)
  # cropping coordinates for x
  cropx0 = max(0, min_x - int(toAdd_x/2))
  cropx1 = min(frame_width-1, max_x + math.ceil(toAdd_x/2))
  if cropx1-cropx0+1 < crop_w and cropx1 == frame_width-1 :
    cropx0 -= crop_w - (cropx1-cropx0+1)
  # cropping coordinates for y
  cropy0 = max(0, min_y - int(toAdd_y/2))
  cropy1 = min(frame_height-1, max_y + math.ceil(toAdd_y/2))
  if cropy1-cropy0+1 < crop_h and cropy1 == frame_height-1 :
    cropy0 -= crop_h - (cropy1-cropy0+1)

  return CroppingSpec(crop_w, crop_h, cropx0, cropy0)

def import_annotations(v7_annotations_file, pos_dir=paths.POS_DIR, class_mapping={'dancing_bee':0}, cropping_spec=None):
    '''
    Process a v7 json file and write position .txt files in format that dataset.py accepts.

    Computes and writes position coordinates & angle from v7 data dict.
    
    Expects V7 annotations to be a polygon or bounding box with directional vector.
    Grabs center of bounding box and offsets if cropping spec is specified.
    Grabs directional vector and converts from [angle from horizontal] to [angle from vertical].
    Writes one .txt file per frame, where each file holds all bee annotations for that frame and each row
    is one bee annotation holding: x center, y center, bee class (0), angle (deg) clockise from the vertical.

    Args:
      v7_annotations_file: full path of json file from v7 application.
      pos_dir: str full path of output dir, which holds sub directories holding .txt files. sub directory name is taken from filename stored in json.
      class_mapping: dict mapping str name of bee class in v7 tool to an integer encoding.
      cropping_spec: cropping specification (of type CroppingSpec), used to offset the bee center coordinates. Expects None if no cropping to the video frame will applied.
    Returns:
      video filename
    '''

    with open(v7_annotations_file, 'r') as f:
      data = json.load(f)
      video_name = get_video_name(data)
      print('Processing json for', video_name, '...')
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
      return get_video_filename(data)

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
        if i==0: print('Original frames shape:',frame.shape)
        # Crop frame.
        if cropping_spec is not None:
            crop_h, crop_w = cropping_spec.height,cropping_spec.width
            xo, yo=cropping_spec.offset_x, cropping_spec.offset_y
            frame = frame[yo:yo+crop_h, xo:xo+crop_w]
            if i==0: print('Cropped frames shape:',frame.shape)
        else:
            if i==0: print('No cropping specified.')
        # Save frame.
        im = Image.fromarray(frame)
        im.save(os.path.join(img_dir, video_name,"%06d.png" % frame_i))

######## MAIN FUNCTION ##############

def import_annotations_and_generate_frames(v7_annotations_file, video_dir, pos_dir=paths.POS_DIR, img_dir=paths.IMG_DIR, crop_w=FR_D, crop_h=FR_D, class_mapping={'dancing_bee':0}):
  cropping_spec = find_cropping_spec(v7_annotations_file, crop_w, crop_h)
  video_filename = import_annotations(v7_annotations_file, pos_dir, class_mapping, cropping_spec)
  video_path = os.path.join(video_dir,video_filename)
  create_frames_from_video(video_path, img_dir=paths.IMG_DIR, frame_range=None, cropping_spec=cropping_spec)