import os
import cv2
from numpy.lib.function_base import average
import argparse
import numpy as np
import os.path as osp

from tqdm import tqdm
from mtcnn.mtcnn import MTCNN

def parse_args():
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    default_haar_cascade_path = 'haarcascade_frontalface_default.xml'
    default_shape = 'rect'
    shape_choices = ['rect', 'circle', 'oval']
    default_resize_factor = -1
    default_detector = 'mtcnn'
    default_distance_threshold = 0.1
    default_time_delta = 1
    
    choices_detector = ['mtcnn', 'haar']

    parser.add_argument('--inpath', type=str, required=True, help='input path (video')
    parser.add_argument('--outpath', type=str, required=True, help='outpath path (video')
    parser.add_argument('--distance_threshold', type=float, default=default_distance_threshold, help='distance threshold for defining closeness')
    parser.add_argument('--time_delta', type=int, default=default_time_delta, help='time delta')
    parser.add_argument('--shape', default=default_shape, choices=shape_choices, help='shape for artifact')
    # parser.add_argument('-rf', '--resize_factor', type=float, default=default_resize_factor, help='resize factor')
    # parser.add_argument('-dt', '--detector', type=str, default=default_detector, choices=choices_detector, help='detector type')
    # parser.add_argument('--haarpath', type=str, default=default_haar_cascade_path, help='path to haar cascade xml')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')
    
    return parser.parse_args()


def get_video_properties(vidcap):
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return length, fps, width, height


def draw_box(face, img, shape, white_color):
    x, y, w, h = face
    if shape == 'rect':
        img[y:y+h, x:x+w] = white_color
    elif shape == 'oval':
        cv2.ellipse(img, (int(x + w // 2), int(y + h // 2)), (w // 2, h // 2), 0, 0, 360, white_color, thickness=-1)
    elif shape == 'circle':
        cv2.circle(img, (x + w // 2, y + h // 2), max(h, w) // 2, white_color, -1)
    else:
        raise Exception('unknown shape: ' + shape)
    
class Box:
    def __init__(self, x, y, w, h):
        
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        self.x1 = x
        self.x2 = x + w
        self.y1 = y
        self.y2 = y + h
    
        self.cx = x + w // 2
        self.cy = y + h // 2
        self.center = (self.cx, self.cy)
        
    def __repr__(self):
        
        return '[{} {} {} {}]'.format(self.x1, self.x2, self.y1, self.y2)
    
    @staticmethod
    def distance(b1, b2):
        return np.sqrt((b1.cx - b2.cx) ** 2 + (b1.cy - b2.cy) ** 2)
    
    @staticmethod
    def average(b1, b2):
        cx = (b1.cx + b2.cx) // 2
        cy = (b1.cy + b2.cy) // 2
        w = (b1.w + b2.w) // 2
        h = (b1.h + b2.h) // 2
        x = cx - w // 2
        y = cy - h // 2
        return Box(x, y, w, h)
    
    def tolist(self):
        return [self.x, self.y, self.w, self.h]
    
    
def detect_faces(img, mtcnn):
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(rgbimg)
    faces = [face['box'] for face in faces]
    return faces
    
if __name__ == '__main__':

    args = parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()
        
    np.random.seed(0)
    
    assert osp.exists(args.inpath), args.inpath + " not found"
    vidcap = cv2.VideoCapture(args.inpath)

    frame_count, fps, width, height = get_video_properties(vidcap)
    
    # if args.resize_factor > 0:
    #     width = int(width * args.resize_factor)
    #     height = int(height * args.resize_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    writer = cv2.VideoWriter(args.outpath, fourcc, fps, (width, height))
    
    white_color = (255, 255, 255)
    mtcnn = MTCNN()
    
    boxes_nearby_times = dict()
    frames_nearby_times = dict()
    
    close = args.distance_threshold * min(height, width)
    
    for nframe in tqdm(range(frame_count)):
        
        success, img = vidcap.read()
        assert success, 'not able to read from video'
        # if args.resize_factor > 0:
        #     img = cv2.resize(img, (width, height))

        
        # detect faces
        frames_nearby_times[nframe] = img
        # what to do when no faces are detected?
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgbimg)
        faces = [face['box'] for face in faces]
        boxes_nearby_times[nframe] = [Box(*face) for face in faces]
        
        prev_prev_t = nframe - 2 * args.time_delta
        prev_t = nframe - args.time_delta
        this_t = nframe
        next_t = nframe + args.time_delta
          
        if prev_t < 0:
            pass          
        elif prev_t >= 0 and prev_prev_t < 0:
            img = frames_nearby_times[prev_t]
            faces = boxes_nearby_times[prev_t]
            faces = [face.tolist() for face in faces]
            for face in faces:
                draw_box(face, img, args.shape, white_color)
            assert img.shape == (height, width, 3), f'img.shape = {img.shape}, height = {height}, width = {width}'
            writer.write(img)
            del frames_nearby_times[prev_t]

        else:
            prev_prev_boxes = boxes_nearby_times[prev_prev_t]
            prev_boxes = boxes_nearby_times[prev_t]    
            this_boxes = boxes_nearby_times[this_t]
            
            # for ppb in prev_prev_boxes:
            #     for tb in this_boxes:
            #         if Box.distance(tb, ppb) < close:
            #             avg_box = Box.average(tb, ppb)
            #             exists = False
            #             for pb in prev_boxes:
            #                 if Box.distance(avg_box, pb) < close:
            #                     exists = True
            #                     break
            #             if not exists:
            #                 prev_boxes.append(avg_box)
            #                 print('filled gap!!!')
            img = frames_nearby_times[prev_t]
            faces = boxes_nearby_times[prev_t]
            faces = [face.tolist() for face in faces]
            if len(faces) != 2:
                pass
                # print(nframe, 'len(faces)', len(faces))
            for face in faces:
                draw_box(face, img, args.shape, white_color)
            assert img.shape == (height, width, 3), f'img.shape = {img.shape}, height = {height}, width = {width}'
            writer.write(img)
            del frames_nearby_times[prev_t]
            
        
    # for nframe in range(frame_count + 1, frame_count + args.time_delta):
    #     this_t = nframe
    #     prev_prev_t = this_t - 2 * args.time_delta
    #     prev_t = this_t - args.time_delta
    #     next_t = this_t + args.time_delta
        
    #     img = frames_nearby_times[prev_t]
    #     faces = boxes_nearby_times[prev_t]
    #     faces = [face.tolist() for face in faces]
    #     for face in faces:
    #         draw_box(face, img, args.shape, white_color)

    #     writer.write(img)
        
    writer.release()