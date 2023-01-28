''' Library to import annotations from V7 software and image frames from video files into a format that dataset.py expects.
'''
import cv2
import json
import math
import numpy as np
import os
from PIL import Image
import plots.plot as plot
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

def find_cropping_spec(v7_annotations_file, crop_w=FR_D, crop_h=FR_D, allowed_instance_ids=None):
  '''
  Given annotations of a video and the crop dimensions, find appropriate cropping offsets so that cropping will include all of the bee's movement.

  v7_annotations_file: full path of json file from v7 application.
  crop_w: width of resulting cropped frame
  crop_h: height of resulting cropped frame
  allowed_instance_ids: List of int instance ids whose bounding boxes will be used to calculate cropping spec. If None, all instances are used.
  '''
  if allowed_instance_ids:
    allowed_instance_ids = set(allowed_instance_ids)
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
        if 'bounding_box' not in frame:
          continue
        if allowed_instance_ids and frame["instance_id"]["value"] not in allowed_instance_ids:
          continue
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
      print()
      return get_video_filename(data)

def create_frames_from_video(video_path, img_dir=paths.IMG_DIR, frame_range=None, cropping_spec=None, fps_to_generate=50):
    '''
    For one video, write frame .png files to subdir under img_dir.

    Crops frames according to cropping spec, if provided.
    

    Args:
      video_path: full path to .mp4 file
      img_dir: output dir to store frames, frames are stored in a sub dir of img_dir called video_name
      frame_range: range of sequential frames numbers (indexed by 0), if empty produces frames for entire video.
      cropping_spec: cropping specification (of type CroppingSpec), used to crop the frame. Expects None if no cropping is applied.
      fps_to_generate: used to determine the proportion of frames to keep. should be <= original fps.
    '''
    video_name = video_path.split('/')[-1].replace('.mp4','')
    print('Processing video', video_name, '...')
    make_dir(os.path.join(img_dir,video_name))
    cap = cv2.VideoCapture(video_path)
    print("total frames",cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_to_generate > original_fps:
        raise ValueError("fps_to_generate: %d is bigger than orignal_fps: %d" % (fps_to_generate, original_fps))

    if frame_range is None:
      frame_range = range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i, frame_i in func.enumerate2(frame_range, step=int(original_fps/fps_to_generate)):
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
    print()

def save_labelled_video(video_filename, output_dir, fps=60, pos_dir=paths.POS_DIR, img_dir=paths.IMG_DIR):
    path_save = os.path.join(output_dir,video_filename)
    video_name = video_filename.replace('.mp4','')
    img_dir=os.path.join(img_dir,video_name)
    frame_files = func.get_all_files([img_dir])
    img_shape = func.read_img(img_file=frame_files[0]).shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path_save, fourcc, fps, img_shape)

    for frame in frame_files:
      img = plot.plot_detections(frame_path=frame, save=False, img_dir=img_dir, pos_dir=os.path.join(pos_dir,video_name), fps=fps)
      img = img.convert("RGB") # Remove 4th layer (transparencny) that is used by some formats like .png
      writer.write(cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR))

    writer.release()

######## MAIN FUNCTION ##############

def import_annotations_and_generate_frames(v7_annotations_file,
                                           video_dir,
                                           pos_dir=paths.POS_DIR, img_dir=paths.IMG_DIR,
                                           crop_w=FR_D, crop_h=FR_D,
                                           class_mapping={'dancing_bee':0},
                                           allowed_instance_ids_for_cropping_spec=None,
                                           frames_range_to_generate=None,
                                           labelled_video_dir=None,
                                           fps_to_generate=25):
  '''

  :param v7_annotations_file: full path of json file from v7 application.
  :param video_dir:
  :param pos_dir:
  :param img_dir:
  :param crop_w:
  :param crop_h:
  :param class_mapping: v7 class name to class index-1.
  :param allowed_instance_ids_for_cropping_spec: list of instance ids to concentrate cropping spec around.
  :param frames_range_to_generate: range of sequential frames numbers (indexed by 0), if empty produces frames for entire video.
  :param labelled_video_dir: if not None, save labelled frames in .mp4 to this dir.
  :param fps_to_generate:  should be <= fps, expected fps of input data to detection model.
  :return: cropping_spec used to cut frames.
  '''
  cropping_spec = find_cropping_spec(v7_annotations_file, crop_w, crop_h, allowed_instance_ids=allowed_instance_ids_for_cropping_spec)
  video_filename = import_annotations(v7_annotations_file, pos_dir, class_mapping, cropping_spec)
  video_path = os.path.join(video_dir,video_filename)
  create_frames_from_video(video_path, img_dir=img_dir, frame_range=frames_range_to_generate, cropping_spec=cropping_spec,fps_to_generate=fps_to_generate)
  if labelled_video_dir:
      save_labelled_video(video_filename, labelled_video_dir, fps=fps_to_generate, pos_dir=pos_dir, img_dir=img_dir)
  return cropping_spec