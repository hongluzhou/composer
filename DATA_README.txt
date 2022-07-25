************************************************************
*** folder 'joints':
This folder contains the HRNet extracted joints of each person of each clip on the Volleyball dataset. 
The folder structure after unzip is:
- video_id
  - clip_id.pickle
  
>>> import pickle
>>> with open('13286.pickle', 'rb') as f:
...     data = pickle.load(f)
... 
>>> type(data)
<class 'dict'>
>>> len(data)
20
>>> data.keys()
dict_keys([13276, 13277, 13278, 13279, 13280, 13281, 13282, 13283, 13284, 13285, 13286, 13287, 13288, 13289, 13290, 13291, 13292, 13293, 13294, 13295])
>>> type(data[13286])
<class 'numpy.ndarray'>
>>> data[13276].shape
(12, 17, 3)
Here, 12 indicates the number of persons per clip in Volleyball.
17 is the number of joints per person.
The last 3 dims are: [x_coord, y_coord, joint_type_class_id].

The mapping between joint type and joint class id is:
COCO_KEYPOINT_INDEXES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}



************************************************************
*** file 'tracks_normalized.pkl' & file 'tracks_normalized_with_person_action_label.pkl':

>>> import pickle
>>> with open('tracks_normalized.pkl', 'rb') as f:
...     tracks = pickle.load(f)
... 
>>> with open('tracks_normalized_with_person_action_label.pkl', 'rb') as f:
...     actions = pickle.load(f)
... 

tracks[(8, 29165)][29166] is a numpy array in shape (N, 4) representing the bounding boxes of the N persons in the video 8 clip 29165 frame 29166 (in Volleyball N is 12).
actions[(8, 29165)][29166] is a numpy array in shape (N, 1) representing the action IDs of the N persons in the video 8 clip 29165 frame 29166.

Please note that tracks[(8, 29165)][29166][i], actions[(8, 29165)][29166][i] and skeleton_this_clip[29166][i] correspond to the same person!
Also, tracks[(8, 29165)][29166][i] and tracks[(8, 29165)][29167][i] correspond to one person since the file is tracked person bounding boxes.
 
The 4-dim based bounding box is normalized [y1, x1, y2, x2]. The following is how I would read the box and visualize the box on a frame (which might help you to understand the format): 
y1,x1,y2,x2 = box
box = [int(x1 * Frame_Width), int(y1 * Frame_Height), int(x2 * Frame_Width), int(y2 * Frame_Height)] 

The following is the mapping between person action class to action ID:
mapping = {
'N/A',: 0,
'blocking': 1,
'digging': 2,
'falling': 3,
'jumping': 4,
'moving': 5,
'setting': 6,
'spiking': 7
'standing': 8,
'waiting': 9
}

************************************************************
*** folder 'volleyball_ball_annotation':
We obtained the ball annotations from:
Perez, Mauricio, Jun Liu, and Alex C. Kot. "Skeleton-based relational reasoning for group activity analysis." Pattern Recognition 122 (2022): 108360.
There is a README.txt inside the folder.

