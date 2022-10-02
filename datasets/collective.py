import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset


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

COCO_KEYPOINT_HORIZONTAL_FLIPPED = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15
}

 
KEYPOINT_PURTURB_RANGE = 1.0



OKS_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


def oks_one_keypoint_compute(keypoint, keypoint_prev, box, box_prev, require_norm=False, OW=720.0, OH=1280.0):
    # Arguments:
    # - keypoint: (3,) - x, y, type 
    # - keypoint_prev: (3,) - x, y, type, this keypoint in previous frame
    # - box: (4,) bounding box of the person
    # - box_previous: (4,), this bounding box in previous frame
    
    keypoint_type = keypoint[2]
    
    if require_norm:
        keypoint[0] /= OH
        keypoint[1] /= OW
        keypoint_prev[0] /= OH
        keypoint_prev[1] /= OW
    
    y1,x1,y2,x2 = box
    if require_norm:
        box = [x1, y1, x2, y2] 
    else:
        box = [x1*OH, y1*OW, x2*OH, y2*OW] 
    area = (box[2]-box[0]) * (box[3]-box[1])
    
    y1,x1,y2,x2 = box_prev
    if require_norm:
        box_prev = [x1, y1, x2, y2] 
    else:
        box_prev = [x1*OH, y1*OW, x2*OH, y2*OW] 
    area_prev = (box_prev[2]-box_prev[0]) * (box_prev[3]-box_prev[1])
    
    avg_area = (area + area_prev) / 2.0
    if avg_area == 0:  # happen when box and keypoint coords are just 0s
        return 0.0
    
    dist = np.linalg.norm(keypoint[:2]-keypoint_prev[:2])

    oks = np.exp(- dist**2 / ( 2 * avg_area * OKS_sigmas[int(keypoint_type)]**2))
    
    return oks


class Collective(Dataset):
    def __init__(self, args, split='train', print_cls_idx=True):
        self.args = args
        self.split = split
        
        self.dataset_splits = {
            'train': [1, 2, 3, 4, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 
                      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            'test': [5,6,7,8,9,10,11,15,16,25,28,29]
        }
        
        
        # activities
        self.idx2class = {
            0: 'Walking',
            1: 'Waiting',
            2: 'Queueing',
            3: 'Talking'
        }
        self.class2idx = dict()
        if print_cls_idx:
            print('class index:') 
        for k in self.idx2class:
            self.class2idx[self.idx2class[k]] = k
            if print_cls_idx:
                print('{}: {}'.format(self.idx2class[k], k))
        self.group_activities_weights = torch.FloatTensor([0.5, 3.0, 2.0, 1.0]).cuda()
        # ACTIVITIES = ['Walking', 'Waiting', 'Queueing', 'Talking']    
                    
        self.person_actions_all = pickle.load(
            open(os.path.join(self.args.dataset_dir, self.args.person_action_label_file_name), "rb"))
        self.person_actions_weights = torch.FloatTensor([0.5, 1.0, 4.0, 3.0, 2.0]).cuda()
        # ACTIONS = ['NA', 'Walking', 'Waiting', 'Queueing', 'Talking']
        
        self.FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}
        
        self.annotations = []
        self.annotations_each_person = []
        self.clip_joints_paths = []
        self.clips = []
        
        self.annotations_CAD = pickle.load(
            open(os.path.join(self.args.dataset_dir, 'annotations.pkl'), "rb"))
        
        
        self.prepare(args.dataset_dir)
            
        if self.args.horizontal_flip_augment and self.split == 'train':
            if self.args.horizontal_flip_augment_purturb:
                self.horizontal_flip_augment_joint_randomness = dict()
                
        if self.args.horizontal_move_augment and self.split == 'train':
            if self.args.horizontal_move_augment_purturb:
                self.horizontal_move_augment_joint_randomness = dict()
                
        if self.args.vertical_move_augment and self.split == 'train':
            if self.args.vertical_move_augment_purturb:
                self.vertical_move_augment_joint_randomness = dict()
            
        if self.args.agent_dropout_augment:
            self.agent_dropout_augment_randomness = dict()
            
        self.collect_standardization_stats()
        
        self.tdata = pickle.load(
            open(os.path.join(self.args.dataset_dir, self.args.tracklets_file_name), "rb")
        )
        
        
    def prepare(self, dataset_dir):
        """
        Prepare the following lists based on the dataset_dir, self.split
            - self.annotations 
            - self.annotations_each_person 
            - self.clip_joints_paths
            - self.clips
            (the following if needed)
            - self.horizontal_flip_mask
            - self.horizontal_mask
            - self.vertical_mask
            - self.agent_dropout_mask
        """  
        clip_joints_paths = []  
        for video in self.dataset_splits[self.split]:
            clip_joints_paths.extend(glob.glob(os.path.join(dataset_dir, self.args.joints_folder_name, str(video), '*.pickle')))
            
        # count = 0
        for path in clip_joints_paths:
            video, clip = path.split('/')[-2], path.split('/')[-1].split('.pickle')[0]
            self.clips.append((video, clip))
            self.annotations.append(self.annotations_CAD[int(video)][int(clip)]['group_activity'])
            self.annotations_each_person.append(self.person_actions_all[(int(video), int(clip))])
            # count += 1
        # print('total number of clips is {}'.format(count))
        
        self.clip_joints_paths += clip_joints_paths
      
        assert len(self.annotations) == len(self.clip_joints_paths)
        assert len(self.annotations) == len(self.annotations_each_person)
        assert len(self.clip_joints_paths) == len(self.clips)
        # pdb.set_trace()

        true_data_size = len(self.annotations)
        true_annotations = copy.deepcopy(self.annotations)
        true_annotations_each_person = copy.deepcopy(self.annotations_each_person)
        true_clip_joints_paths = copy.deepcopy(self.clip_joints_paths)
        true_clips = copy.deepcopy(self.clips)
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.horizontal_flip_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
                
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
      
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            
        # if random agent dropout augmentation and is training
        if self.args.agent_dropout_augment and self.split == 'train':
            self.agent_dropout_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            
    
    def __len__(self):
        return len(self.clip_joints_paths)
    
    
    def personwise_normalization(self, points, offset_points=None, 
                                 scale_distance_point_pairs=None,
                                 scale_distance_fn=None, scale_unit=None,
                                 video=None, clip=None, frame=None):
        '''
        https://github.com/google-research/google-research/blob/8b51f2bcf6799d4885ce5997ed5a5111110c67db/poem/core/keypoint_profiles.py#L511
        
        Args:
            points: A tensor for points. Shape = [..., num_points, point_dim].
            offset_points: A list of points to compute the center point. 
                  If a single point is specified, the point will be
                  used as the center point. If multiple points are specified, the centers of their
                  corresponding coordinates will be used as the center point.
            scale_distance_point_pairs: A list of tuples and each tuple is a pair of point lists. 
                  The points will be converted into point indices later. Then, for
                  example, use [([0], [1]), ([2, 3], [4])] to compute scale distance as the
                  scale_distance_fn of the distance between point_0 and point_1 and the distance
                  between the center(point_2, point_3) and point_4.
            scale_distance_fn: A  function handle for distance, e.g., torch.sum.
            scale_unit: A scalar for the scale unit whose value the scale distance will
                  be scaled to.
        '''
        if not offset_points:
            offset_points = ['left_hip', 'right_hip']
        if not scale_distance_point_pairs:
            scale_distance_point_pairs = [(['left_shoulder'], ['right_shoulder']),
                                          (['left_shoulder'], ['left_hip']),
                                          (['left_shoulder'], ['right_hip']),
                                          (['right_shoulder'], ['left_hip']),
                                          (['right_shoulder'], ['right_hip']),
                                          (['left_hip'], ['right_hip'])]
        if not scale_distance_fn:
            scale_distance_fn = 'max'  # {sum, max, mean}
        if not scale_unit:
            scale_unit = 1.0
        # https://github.com/google-research/google-research/blob/8b51f2bcf6799d4885ce5997ed5a5111110c67db/poem/core/keypoint_profiles.py#L950
        
        # get indices instead of keypoint names
        offset_point_indices = [COCO_KEYPOINT_INDEXES[keypoint] for keypoint in offset_points]
        scale_distance_point_index_pairs = []
        for (points_i, points_j) in scale_distance_point_pairs:
            points_i_indices = []
            for keypoint in points_i:
                points_i_indices.append(COCO_KEYPOINT_INDEXES[keypoint])
                
            points_j_indices = []
            for keypoint in points_j:
                points_j_indices.append(COCO_KEYPOINT_INDEXES[keypoint])
            
            scale_distance_point_index_pairs.append((points_i_indices, points_j_indices))
            
        def get_points(points, indices):
            """Gets points as the centers of points at specified indices.

              Args:
                points: A tensor for points. Shape = [..., num_points, point_dim].
                indices: A list of integers for point indices.

              Returns:
                A tensor for (center) points. Shape = [..., 1, point_dim].

              Raises:
                ValueError: If `indices` is empty.
            """
            if not indices:
                raise ValueError('`Indices` must be non-empty.')
            points = points[:, indices, :]
            if len(indices) == 1:
                return points
            return torch.mean(points, dim=-2, keepdim=True)
        
        offset_point = get_points(points, offset_point_indices)
        
        def compute_l2_distances(lhs, rhs, squared=False, keepdim=False):
            """Computes (optionally squared) L2 distances between points.

              Args:
                lhs: A tensor for LHS points. Shape = [..., point_dim].
                rhs: A tensor for RHS points. Shape = [..., point_dim].
                squared: A boolean for whether to compute squared L2 distance instead.
                keepdims: A boolean for whether to keep the reduced `point_dim` dimension
                  (of length 1) in the result distance tensor.

              Returns:
                A tensor for L2 distances. Shape = [..., 1] if `keepdims` is True, or [...]
                  otherwise.
            """
            squared_l2_distances = torch.sum(
                (lhs - rhs)**2, dim=-1, keepdim=keepdim)
            return torch.sqrt(squared_l2_distances) if squared else squared_l2_distances
        
        def compute_scale_distances():
            sub_scale_distances_list = []
            for lhs_indices, rhs_indices in scale_distance_point_index_pairs:
                lhs_points = get_points(points, lhs_indices)  # Shape = [..., 1, point_dim]
                rhs_points = get_points(points, rhs_indices)  # Shape = [..., 1, point_dim]
                sub_scale_distances_list.append(
                    compute_l2_distances(lhs_points, rhs_points, squared=True, keepdim=True))  # Euclidean distance
               
            sub_scale_distances = torch.cat(sub_scale_distances_list, dim=-1)
            
            if scale_distance_fn == 'sum':
                return torch.sum(sub_scale_distances, dim=-1, keepdim=True)
            elif scale_distance_fn == 'max':
                return torch.max(sub_scale_distances, dim=-1, keepdim=True).values
            elif scale_distance_fn == 'mean':
                return torch.mean(sub_scale_distances, dim=-1, keepdim=True)
            else:
                print('Please check whether scale_distance_fn is supported!')
                os._exit(0)
        
        scale_distances = compute_scale_distances()
        
        normalized_points = (points - offset_point) / (scale_distances * scale_unit + 1e-12)  # in case divide by 0
        return normalized_points

            
    def collect_standardization_stats(self):
        # get joint x/y mean, std over train set
        if self.split == 'train':
            if self.args.recollect_stats_train or (
                not os.path.exists(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'))):
                joint_xcoords = []
                joint_ycoords = []
                joint_dxcoords = []
                joint_dycoords = [] 

                for index in range(self.__len__()):   # including augmented data!
                    with open(self.clip_joints_paths[index] , 'rb') as f:
                        joint_raw = pickle.load(f)
                        
                    frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]
                    
                    # if horizontal flip augmentation and is training
                    if self.args.horizontal_flip_augment:
                        if index < len(self.horizontal_flip_mask):
                            if self.horizontal_flip_mask[index]:
                                if self.args.horizontal_flip_augment_purturb:
                                    self.horizontal_flip_augment_joint_randomness[index] = defaultdict()
                                    joint_raw = self.horizontal_flip_augment_joint(
                                        joint_raw, frames, 
                                        add_purturbation=True, randomness_set=False, index=index, video_id=int(video))
                                else:
                                    joint_raw = self.horizontal_flip_augment_joint(joint_raw, frames, video_id=int(video))
                                    
                    # if horizontal move augmentation and is training
                    if self.args.horizontal_move_augment:
                        if index < len(self.horizontal_mask):
                            if self.horizontal_mask[index]:
                                if self.args.horizontal_move_augment_purturb:
                                    self.horizontal_move_augment_joint_randomness[index] = defaultdict()
                                    joint_raw = self.horizontal_move_augment_joint(
                                        joint_raw, frames,  
                                        add_purturbation=True, randomness_set=False, index=index)
                                else:
                                    joint_raw = self.horizontal_move_augment_joint(joint_raw, frames)
                            
                    # if vertical move augmentation and is training
                    if self.args.vertical_move_augment:
                        if index < len(self.vertical_mask):
                            if self.vertical_mask[index]:
                                if self.args.vertical_move_augment_purturb:
                                    self.vertical_move_augment_joint_randomness[index] = defaultdict()
                                    joint_raw = self.vertical_move_augment_joint(
                                        joint_raw, frames,  
                                        add_purturbation=True, randomness_set=False, index=index)
                                else:
                                    joint_raw = self.vertical_move_augment_joint(joint_raw, frames)
                                    
                    # To compute statistics, no need to consider the random agent dropout augmentation,
                    # but we can set the randomness here.
                    # if random agent dropout augmentation and is training
                    if self.args.agent_dropout_augment:
                        if index < len(self.agent_dropout_mask):
                            if self.agent_dropout_mask[index]:
                                # chosen_frame = random.choice(frames)
                                chosen_person = random.choice(range(self.args.N))
                                # self.agent_dropout_augment_randomness[index] = (chosen_frame, chosen_person)
                                self.agent_dropout_augment_randomness[index] = chosen_person
                    
                    
                    (video, clip) = self.clips[index]
                    joint_raw = self.joints_sanity_fix(joint_raw, frames, video_id=int(video))
                    

                    for tidx, frame in enumerate(frames):
                        joint_xcoords.extend(joint_raw[frame][:,:,0].flatten().tolist())
                        joint_ycoords.extend(joint_raw[frame][:,:,1].flatten().tolist())

                        if tidx != 0:
                            pre_frame = frames[tidx-1]
                            joint_dxcoords.extend((joint_raw[frame][:,:,0]-joint_raw[pre_frame][:,:,0]).flatten().tolist())
                            joint_dycoords.extend((joint_raw[frame][:,:,1]-joint_raw[pre_frame][:,:,1]).flatten().tolist())
                        else:
                            joint_dxcoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            joint_dycoords.extend((np.zeros((self.args.N, self.args.J))).flatten().tolist())
                            
                    
                # -- collect mean std
                joint_xcoords_mean, joint_xcoords_std = np.mean(joint_xcoords), np.std(joint_xcoords)
                joint_ycoords_mean, joint_ycoords_std = np.mean(joint_ycoords), np.std(joint_ycoords)
                joint_dxcoords_mean, joint_dxcoords_std = np.mean(joint_dxcoords), np.std(joint_dxcoords)
                joint_dycoords_mean, joint_dycoords_std = np.mean(joint_dycoords), np.std(joint_dycoords) 

                self.stats = {
                    'joint_xcoords_mean': joint_xcoords_mean, 'joint_xcoords_std': joint_xcoords_std,
                    'joint_ycoords_mean': joint_ycoords_mean, 'joint_ycoords_std': joint_ycoords_std,
                    'joint_dxcoords_mean': joint_dxcoords_mean, 'joint_dxcoords_std': joint_dxcoords_std,
                    'joint_dycoords_mean': joint_dycoords_mean, 'joint_dycoords_std': joint_dycoords_std
                }
                    
                    
                os.makedirs(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name), exist_ok=True)
                with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'wb') as f:
                    pickle.dump(self.stats, f)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_flip_augment_joint_randomness, f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.horizontal_move_augment_joint_randomness, f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'vertical_move_augment_joint_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.vertical_move_augment_joint_randomness, f)
                        
                if self.args.agent_dropout_augment:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'agent_dropout_augment_randomness.pickle'), 'wb') as f:
                        pickle.dump(self.agent_dropout_augment_randomness, f)
                    
            else:
                try:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'rb') as f:
                        self.stats = pickle.load(f)
                except FileNotFoundError:
                    print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                    os._exit(0)
                    
                if self.args.horizontal_flip_augment and self.args.horizontal_flip_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_flip_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_flip_augment_joint_randomness = pickle.load(f)
                        
                if self.args.horizontal_move_augment and self.args.horizontal_move_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'horizontal_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.horizontal_move_augment_joint_randomness = pickle.load(f)
                        
                if self.args.vertical_move_augment and self.args.vertical_move_augment_purturb:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'vertical_move_augment_joint_randomness.pickle'), 'rb') as f:
                        self.vertical_move_augment_joint_randomness = pickle.load(f)
                
                if self.args.agent_dropout_augment:
                    with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 
                                           'agent_dropout_augment_randomness.pickle'), 'rb') as f:
                        self.agent_dropout_augment_randomness = pickle.load(f)
        else:
            try:
                with open(os.path.join('datasets', self.args.dataset_name, self.args.joints_folder_name, 'stats_train.pickle'), 'rb') as f:
                    self.stats = pickle.load(f)
            except FileNotFoundError:
                print('Dataset statistics (e.g., mean, std) are missing! The dataset statistics pickle file should be generated during training.')
                os._exit(0)
                
                
    def joints_sanity_fix(self, joint_raw, frames, video_id=None):
        # note that it is possible the width_coords>1280 and height_coords>720 due to imperfect pose esitimation
        # here we fix these cases
        
        for t in joint_raw:
            for n in range(len(joint_raw[t])):
                for j in range(len(joint_raw[t][n])):
                    # joint_raw[t][n, j, 0] = int(joint_raw[t][n, j, 0])
                    # joint_raw[t][n, j, 1] = int(joint_raw[t][n, j, 1])
                    
                    if video_id != None:
                        if joint_raw[t][n, j, 0] >= self.FRAMES_SIZE[video_id][1]:
                            joint_raw[t][n, j, 0] = self.FRAMES_SIZE[video_id][1] - 1

                        if joint_raw[t][n, j, 1] >= self.FRAMES_SIZE[video_id][0]:
                            joint_raw[t][n, j, 1] = self.FRAMES_SIZE[video_id][0] - 1

                        if joint_raw[t][n, j, 0] < 0:
                            joint_raw[t][n, j, 0] = 0

                        if joint_raw[t][n, j, 1] < 0:
                            joint_raw[t][n, j, 1] = 0 
                    else:
                        if joint_raw[t][n, j, 0] >= self.args.image_w:
                            joint_raw[t][n, j, 0] = self.args.image_w - 1

                        if joint_raw[t][n, j, 1] >= self.args.image_h:
                            joint_raw[t][n, j, 1] = self.args.image_h - 1

                        if joint_raw[t][n, j, 0] < 0:
                            joint_raw[t][n, j, 0] = 0

                        if joint_raw[t][n, j, 1] < 0:
                            joint_raw[t][n, j, 1] = 0 
                        
        # modify joint_raw - loop over each frame and pad the person dim because it can have less than N persons
        for f in joint_raw:
            n_persons = joint_raw[f].shape[0]
            if n_persons < self.args.N:  # padding in case some clips has less than N persons
                joint_raw[f] = np.concatenate((
                    joint_raw[f], 
                    np.zeros((self.args.N-n_persons, self.args.J, joint_raw[f].shape[2]))), 
                    axis=0)
        return joint_raw
    
    
    def horizontal_flip_augment_joint(
        self, joint_raw, frames, add_purturbation=False, randomness_set=False, index=0, video_id=None):
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    if video_id != None:
                        joint_raw[t][n, j, 0] = self.FRAMES_SIZE[video_id][1] - joint_raw[t][n, j, 0]  # flip joint coordinates
                    else:
                        joint_raw[t][n, j, 0] = self.args.image_w - joint_raw[t][n, j, 0]  # flip joint coordinates
                    if add_purturbation:
                        if not randomness_set:
                            self.horizontal_flip_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 0] += self.horizontal_flip_augment_joint_randomness[index][(t, n, j)]
                    joint_raw[t][n, j, 2] = COCO_KEYPOINT_HORIZONTAL_FLIPPED[joint_raw[t][n, j, 2]]  # joint class type has to be flipped
                joint_raw[t][n] = joint_raw[t][n][joint_raw[t][n][:, 2].argsort()]  # sort by joint type class id
        return joint_raw
    
    
    def horizontal_move_augment_joint(
        self, joint_raw, frames, add_purturbation=False, randomness_set=True, index=0, max_horizontal_diff=10.0, ball_trajectory=None):
        horizontal_change = np.random.uniform(low=-max_horizontal_diff, high=max_horizontal_diff)
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 0] += horizontal_change  # horizontally move joint 
                    if add_purturbation:
                        if not randomness_set:
                            self.horizontal_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 0] += self.horizontal_move_augment_joint_randomness[index][(t, n, j)]
        if ball_trajectory is not None:
            for t in range(len(ball_trajectory)):
                 ball_trajectory[t, 0] += horizontal_change
            return joint_raw, ball_trajectory
        else:
            return joint_raw
        
    
    def vertical_move_augment_joint(
        self, joint_raw, frames, add_purturbation=False, randomness_set=True, index=0, max_vertical_diff=10.0, ball_trajectory=None):
        vertical_change = np.random.uniform(low=-max_vertical_diff, high=max_vertical_diff)
        for t in frames:
            for n in range(len(joint_raw[t])):
                if not np.any(joint_raw[t][n][:,:2]):  # all 0s, not actual joint coords
                    continue
                for j in range(len(joint_raw[t][n])):
                    joint_raw[t][n, j, 1] += vertical_change  # vertically move joint 
                    if add_purturbation:
                        if not randomness_set:
                            self.vertical_move_augment_joint_randomness[index][(t, n, j)] = random.uniform(
                                -KEYPOINT_PURTURB_RANGE, KEYPOINT_PURTURB_RANGE)
                        joint_raw[t][n, j, 1] += self.vertical_move_augment_joint_randomness[index][(t, n, j)]
        if ball_trajectory is not None:
            for t in range(len(ball_trajectory)):
                 ball_trajectory[t, 1] += vertical_change
            return joint_raw, ball_trajectory
        else:
            return joint_raw


    def agent_dropout_augment_joint(self, joint_feats, frames, index=0, J=17):
        # joint_feats: (N, J, T, d)
        chosen_person = self.agent_dropout_augment_randomness[index]
        T = joint_feats.shape[2]
        feature_dim = joint_feats.shape[3]

        joint_feats[chosen_person, :, :, :] = torch.zeros(J, T, feature_dim)
        return joint_feats
    
    
    def __getitem__(self, index):
        # index = 0
        current_joint_feats_path = self.clip_joints_paths[index] 
        (video, clip) = self.clips[index]
        label = self.annotations[index]
        person_labels = self.annotations_each_person[index]
        
        joint_raw = pickle.load(open(current_joint_feats_path, "rb"))
        # joint_raw: T: (N, J, 3)
        # 3: [joint_x, joint_y, joint_type]
        
        frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]
                        
        person_labels = torch.LongTensor(person_labels[frames[0]].squeeze())  # person action remains to be the same across all frames 
        # person_labels: (N, )
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            if index < len(self.horizontal_flip_mask):
                if self.horizontal_flip_mask[index]:
                    if self.args.horizontal_flip_augment_purturb:
                        joint_raw = self.horizontal_flip_augment_joint(
                            joint_raw, frames, add_purturbation=True, randomness_set=True, index=index, video_id=int(video))
                    else:
                        joint_raw = self.horizontal_flip_augment_joint(joint_raw, frames, video_id=int(video))
                            
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            if index < len(self.horizontal_mask):
                if self.horizontal_mask[index]:
                    if self.args.horizontal_move_augment_purturb:
                        joint_raw = self.horizontal_move_augment_joint(
                            joint_raw, frames, add_purturbation=True, randomness_set=True, index=index)
                    else:
                        joint_raw = self.horizontal_move_augment_joint(joint_raw, frames)  
                    
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            if index < len(self.vertical_mask):
                if self.vertical_mask[index]:
                    if self.args.vertical_move_augment_purturb:
                        joint_raw = self.vertical_move_augment_joint(
                            joint_raw, frames, add_purturbation=True, randomness_set=True, index=index)
                    else:
                        joint_raw = self.vertical_move_augment_joint(joint_raw, frames)                  
                    
        joint_raw = self.joints_sanity_fix(joint_raw, frames, video_id=int(video))
        
        
        # get joint_coords_all for image coordinates embdding
        if self.args.image_position_embedding_type != 'None':
            joint_coords_all = []
            for n in range(self.args.N):
                joint_coords_n = []

                for j in range(self.args.J):
                    joint_coords_j = []

                    for tidx, frame in enumerate(frames):
                        joint_x, joint_y, joint_type = joint_raw[frame][n,j,:]
                        
                        # joint_x = min(int(joint_x), self.args.image_w-1)
                        # joint_y = min(int(joint_y), self.args.image_h-1)
                        joint_x = min(joint_x, self.FRAMES_SIZE[int(video)][1]-1)
                        joint_y = min(joint_y, self.FRAMES_SIZE[int(video)][0]-1)
                        joint_x = max(0, joint_x)
                        joint_y = max(0, joint_y)

                        joint_coords = []
                        joint_coords.append(joint_x)  # width axis 
                        joint_coords.append(joint_y)  # height axis
                            
                        joint_coords_j.append(joint_coords)
                    joint_coords_n.append(joint_coords_j)   
                joint_coords_all.append(joint_coords_n)
                
                
        # get basic joint features (std normalization)
        joint_feats_basic = []  # (N, J, T, d_0_v1)

        for n in range(self.args.N):
            joint_feats_n = []

            for j in range(self.args.J):
                joint_feats_j = []

                for tidx, frame in enumerate(frames):
                    joint_x, joint_y, joint_type = joint_raw[frame][n,j,:]

                    joint_feat = []

                    joint_feat.append((joint_x-self.stats['joint_xcoords_mean'])/self.stats['joint_xcoords_std'])
                    joint_feat.append((joint_y-self.stats['joint_ycoords_mean'])/self.stats['joint_ycoords_std'])

                    if tidx != 0:
                        pre_frame = frames[tidx-1] 
                        pre_joint_x, pre_joint_y, pre_joint_type = joint_raw[pre_frame][n,j,:]
                        joint_dx, joint_dy = joint_x - pre_joint_x, joint_y - pre_joint_y 
                    else:
                        joint_dx, joint_dy = 0, 0

                    joint_feat.append((joint_dx-self.stats['joint_dxcoords_mean'])/self.stats['joint_dxcoords_std'])
                    joint_feat.append((joint_dy-self.stats['joint_dycoords_mean'])/self.stats['joint_dycoords_std'])
                    joint_feats_j.append(joint_feat)
                joint_feats_n.append(joint_feats_j)
            joint_feats_basic.append(joint_feats_n)
                
            
        # person-wise normalization
        joint_feats_advanced = []  # (N, J, T, d_0_v2)

        joint_personwise_normalized = dict()
        for f in frames:
            joint_personwise_normalized[f] = self.personwise_normalization(
                torch.Tensor(joint_raw[f][:,:,:-1]), 
                video=video, clip=clip, frame=f)  
            # (N, J, 2) the 2 dims are joint_x, joint_y

        for n in range(self.args.N):
            joint_feats_n = []

            for j in range(self.args.J):
                joint_feats_j = []

                for frame in frames:
                    joint_x, joint_y = joint_personwise_normalized[frame][n,j,:]
                    joint_type = joint_raw[frame][n,j,-1]

                    joint_feat = []
                    joint_feat.append(joint_x)
                    joint_feat.append(joint_y)
                    joint_feat.append(int(joint_type))  # start from 0

                    joint_feats_j.append(joint_feat)
                joint_feats_n.append(joint_feats_j)
            joint_feats_advanced.append(joint_feats_n)
            
        # adding joint metric features
        joint_feats_metrics = []  # (N, J, T, d_metrics)
        for frame_idx, frame in enumerate(frames):  # loop over frames of this clip
            this_frame_metric_scores = []
            for player_idx in range(self.args.N):  # loop over players
                this_player_metric_scores = []
                for joint_idx in range(self.args.J):  # loop over joints
                    if frame_idx == 0:  # first frame
                        this_player_metric_scores.append([1.0])
                    else:
                        frame_previous = frames[frame_idx-1]
                        OKS_score = oks_one_keypoint_compute(
                            joint_raw[frame][player_idx,joint_idx,:],
                            joint_raw[frame_previous][player_idx,joint_idx,:],
                            self.tdata[(int(video), int(clip))][frame][player_idx],
                            self.tdata[(int(video), int(clip))][frame_previous][player_idx]
                            )
                        this_player_metric_scores.append([OKS_score])
                this_frame_metric_scores.append(this_player_metric_scores)
            joint_feats_metrics.append(this_frame_metric_scores)
        joint_feats_metrics = np.array(joint_feats_metrics) # (T, N, J, 2) or  # (T, N, J, 1)

        # mean aggregate by a person's joints
        person_agg = np.mean(joint_feats_metrics, axis=2)
        joint_feats_metrics = np.concatenate(
            (joint_feats_metrics,
             np.tile(person_agg, self.args.J)[:,:,:,np.newaxis]), axis=-1)

        
        
        joint_feats = torch.cat((torch.Tensor(np.array(joint_feats_basic)), 
                                 torch.Tensor(np.array(joint_feats_metrics)).permute(1,2,0,3), 
                                 torch.Tensor(np.array(joint_feats_advanced)), 
                                 torch.Tensor(np.array(joint_coords_all))), dim=-1)  
        
        
        # if random agent dropout augmentation and is training                
        if self.args.agent_dropout_augment and self.split == 'train':
            if index < len(self.agent_dropout_mask):
                if self.agent_dropout_mask[index]:
                    joint_feats = self.agent_dropout_augment_joint(
                            joint_feats, frames, index=index, J=self.args.J)
                    
        ball_feats = torch.zeros(len(frames), 6)
            
        
        assert not torch.isnan(joint_feats).any() 
        
        return joint_feats, label, video, clip, person_labels, ball_feats
 