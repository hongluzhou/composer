import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from kmeans_pytorch import kmeans

from models.misc import build_mlp 
from models.misc import GraphConvolution, get_joint_graph
from models.position_encoding import PositionEmbeddingAbsoluteLearned_1D
from models.position_encoding import LearnedFourierFeatureTransform

import pdb


class COMPOSER(nn.Module):
    def __init__(self, args):
        super(COMPOSER, self).__init__()
        
        self.args = args
        
        self.interaction_indexes = [
            self.args.N*i+j for i in range(self.args.N) 
            for j in range(self.args.N) if self.args.N*i+j != self.args.N*i+i]
         
        embedding_dim = args.joint_initial_feat_dim
        self.joint_class_embed_layer = nn.Embedding(args.J, embedding_dim)
        gcn_layers = [
            GraphConvolution(in_features=embedding_dim, out_features=embedding_dim,
                             dropout=0, act=F.relu, use_bias=True) 
            for l in range(self.args.num_gcn_layers)] 
        self.joint_class_gcn_layers = nn.Sequential(*gcn_layers)

        self.adj = get_joint_graph(num_nodes=args.J, joint_graph_path='models/joint_graph.txt')
        
        self.special_token_embed_layer = nn.Embedding(args.max_num_tokens, args.TNT_hidden_dim)
        self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(args.max_times_embed, embedding_dim)
        self.image_embed_layer = LearnedFourierFeatureTransform(2, embedding_dim // 2)
        
        # joint track projection layer
        self.joint_track_projection_layer = build_mlp(input_dim=args.T*embedding_dim*4, 
                                                hidden_dims=[args.TNT_hidden_dim], 
                                                output_dim=args.TNT_hidden_dim,
                                                use_batchnorm=args.projection_batchnorm,
                                                dropout=args.projection_dropout)

        # person track projection layer
        self.person_track_projection_layer = build_mlp(input_dim=args.J*args.T*self.args.joint2person_feat_dim,
                                                 hidden_dims=[args.TNT_hidden_dim], 
                                                 output_dim=args.TNT_hidden_dim,
                                                 use_batchnorm=args.projection_batchnorm,
                                                 dropout=args.projection_dropout)
        
        # person interaction track projection layer
        self.interaction_track_projection_layer = build_mlp(input_dim=args.TNT_hidden_dim*2,
                                                      hidden_dims=[args.TNT_hidden_dim], 
                                                      output_dim=args.TNT_hidden_dim,
                                                      use_batchnorm=args.projection_batchnorm,
                                                      dropout=args.projection_dropout)
        
        # group track projection layer
        if self.args.dataset_name == 'volleyball':
            self.person_to_group_projection = build_mlp(input_dim=(args.N//2)*args.TNT_hidden_dim,
                                                      hidden_dims=[args.TNT_hidden_dim], 
                                                      output_dim=args.TNT_hidden_dim,
                                                      use_batchnorm=args.projection_batchnorm,
                                                      dropout=args.projection_dropout)
  
        
        # ball track projection layer
        if hasattr(args, 'ball_trajectory_use') and args.ball_trajectory_use:
            self.ball_track_projection_layer = build_mlp(input_dim=(2*args.joint_initial_feat_dim+4)*args.T,
                                                     hidden_dims=[args.TNT_hidden_dim], 
                                                     output_dim=args.TNT_hidden_dim,
                                                     use_batchnorm=args.projection_batchnorm,
                                                     dropout=args.projection_dropout)
        
        
        # TNT blocks
        if hasattr(args, 'ball_trajectory_use') and args.ball_trajectory_use:
            from models.tnt_four_scales_with_ball import TNT
        else:
            if self.args.dataset_name == 'volleyball':
                from models.tnt_four_scales import TNT
            elif self.args.dataset_name == 'collective':
                from models.tnt_four_scales_collective import TNT
            else:
                print('Please check the dataset name!')
                os._exit(0)
            
        self.TNT = TNT(args, args.TNT_hidden_dim, args.TNT_n_layers, final_norm=True, return_intermediate=True)
        
        
        # Prediction
        self.classifier = build_mlp(input_dim=args.TNT_hidden_dim, 
                                    hidden_dims=None, output_dim=args.num_classes, 
                                    use_batchnorm=args.classifier_use_batchnorm, 
                                    dropout=args.classifier_dropout)
        
        self.person_classifier = build_mlp(input_dim=args.TNT_hidden_dim, 
                                    hidden_dims=None, output_dim=args.num_person_action_classes, 
                                    use_batchnorm=args.classifier_use_batchnorm, 
                                    dropout=args.classifier_dropout)
            
        # Prototypes
        self.prototypes = nn.Linear(args.TNT_hidden_dim, args.nmb_prototypes, bias=False)
        
        
        
         
    def forward(self, joint_feats_thisbatch, ball_feats_thisbatch):
        
        B = joint_feats_thisbatch.size(0)
        N = joint_feats_thisbatch.size(1)
        J = joint_feats_thisbatch.size(2)
        T = joint_feats_thisbatch.size(3)
        
        d = self.args.TNT_hidden_dim
        
        device = joint_feats_thisbatch.device
        
        joint_feats_for_person_thisbatch = joint_feats_thisbatch[:,:,:,:,:self.args.joint2person_feat_dim]
        joint_img_coords = joint_feats_thisbatch[:,:,:,:,-2:]
          
        # image coords positional encoding
        image_coords = joint_feats_thisbatch[:,:,:,:,-2:].to(torch.int64).cuda()
        coords_h = np.linspace(0, 1, self.args.image_h, endpoint=False)
        coords_w = np.linspace(0, 1, self.args.image_w, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
        image_coords_learned =  self.image_embed_layer(xy_grid).squeeze(0).permute(1, 2, 0)
        image_coords_embeded = image_coords_learned[image_coords[:,:,:,:,1], image_coords[:,:,:,:,0]]
        # (B, N, J, T, d_0)
        
        # update by joint_feats_thisbatch removing the raw joint coordinates dim (last 2 dims by default)
        joint_feats_thisbatch = joint_feats_thisbatch[:,:,:,:,:-2]  

            
        # time positional encoding
        time_ids = torch.arange(1, T+1, device=device).repeat(B, N, J, 1)
        time_seq = self.time_embed_layer(time_ids) 
        
        # joint classes embedding learning as tokens/nodes
        joint_class_ids = joint_feats_thisbatch[:,:,:,:,-1]  # note that the last dim is the joint class id by default
        joint_classes_embeded = self.joint_class_embed_layer(joint_class_ids.type(torch.LongTensor).cuda()) # (B, N, J, T, d_0)
        
        x = joint_classes_embeded.transpose(2, 3).flatten(0, 1).flatten(0, 1)  # x: (B*N*T, J, d_0)
        input = (x, self.adj.repeat(B*N*T, 1, 1).cuda())  # adj: # (B*N*T, J, J)
        joint_classes_encode = self.joint_class_gcn_layers(input)[0]
        joint_classes_encode = joint_classes_encode.view(B, N, T, J, -1).transpose(2, 3)  # (B, N, J, T, d_0)
         
        # update by joint_feats_thisbatch removing the joint class dim (last dim by default)
        joint_feats_thisbatch = joint_feats_thisbatch[:,:,:,:,:-1]
            

        # CLS initial embedding
        CLS_id = torch.arange(1, device=device).repeat(B, 1)
        CLS = self.special_token_embed_layer(CLS_id)

        
        joint_feats_composite_encoded = torch.cat(
            [joint_feats_thisbatch, time_seq, image_coords_embeded, joint_classes_encode], 
            dim=-1) 
        
        
        # PROJECTIONS
        # joint track projection
        joint_track_feats_thisbatch_proj = self.joint_track_projection_layer(
            joint_feats_composite_encoded.flatten(3, 4).flatten(0, 1).flatten(0, 1)  # (B*N*J, T*d_0)
        ).view(B, N*J, -1)
        # (B, N*J, d)
        
        # person track projection
        person_track_feats_thisbatch_proj = self.person_track_projection_layer(
            joint_feats_for_person_thisbatch.flatten(0, 1).contiguous().view(B*N, -1)
        ).view(B, N, -1)
        # (B, N, d)
        
        # form sequence of person-person-interaction-track tokens
        tem1 = person_track_feats_thisbatch_proj.repeat(1, N, 1).reshape(B,N,N,d).transpose(1, 2).flatten(1, 2)  # (B, N^2, d)
        tem2 = person_track_feats_thisbatch_proj.repeat(1, N, 1) # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interaction_track_feats_thisbatch = tem3[:, self.interaction_indexes, :]  # (B, N*(N-1), 2*d)
        interaction_track_feats_thisbatch_proj = self.interaction_track_projection_layer(
            interaction_track_feats_thisbatch.flatten(0, 1)).view(B, N*(N-1), -1)  # (B, N*(N-1), d)
        
        
        # obtain person to group mapping
        if self.args.dataset_name == 'volleyball':
            people_middle_hip_coords = (
                joint_feats_for_person_thisbatch[:,:,11,self.args.group_person_frame_idx,-2:] + 
                joint_feats_for_person_thisbatch[:,:,12,self.args.group_person_frame_idx,-2:]) / 2
            # (B, N, 2)  - W, H (X, Y)

            people_idx_sort_by_middle_hip_xcoord = torch.argsort(people_middle_hip_coords[:,:,0], dim=-1)  # (B, N)
            left_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, :int(self.args.N//2)]  # (B, N/2)
            right_group_people_idx = people_idx_sort_by_middle_hip_xcoord[:, int(self.args.N//2):]  # (B, N/2)
      
            # form sequence of group track tokens
            batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, N // 2)
            left_group_people_repre = person_track_feats_thisbatch_proj[batch_indices, left_group_people_idx]
            right_group_people_repre = person_track_feats_thisbatch_proj[batch_indices, right_group_people_idx]
            left_group_feats_thisbatch_proj = self.person_to_group_projection(left_group_people_repre.flatten(1,2))   # (B, d)
            right_group_feats_thisbatch_proj = self.person_to_group_projection(right_group_people_repre.flatten(1,2))   # (B, d)
            group_track_feats_thisbatch_proj = torch.stack([left_group_feats_thisbatch_proj, right_group_feats_thisbatch_proj], dim=1)  # (B, 2, d)
        
        elif self.args.dataset_name == 'collective':
            cluster_iterations = 100
            while cluster_iterations == 100:
                cluster_ids_x, cluster_centers, cluster_iterations = kmeans(
                    X=person_track_feats_thisbatch_proj.flatten(0,1), num_clusters=self.args.G, distance='cosine', 
                    tqdm_flag=False, iter_limit=100, device=device)
            cluster_ids_x = cluster_ids_x.view(person_track_feats_thisbatch_proj.shape[0],-1)  # (B, N)
            
            group_track_feats_thisbatch_proj = torch.zeros((B, 2, d), device=device)
            for b_idx in range(person_track_feats_thisbatch_proj.shape[0]):
                for c_idx in range(self.args.G):
                    this_cluster_person_idx = (cluster_ids_x[b_idx] == c_idx).nonzero().squeeze()
                    group_track_feats_thisbatch_proj[b_idx, c_idx] =  torch.sum(
                        person_track_feats_thisbatch_proj[b_idx, this_cluster_person_idx, :], dim=0)
                
        else:
            print('Please check the dataset name!')
            os._exit(0)
               
        # ball related encodings
        if hasattr(self.args, 'ball_trajectory_use') and self.args.ball_trajectory_use:
            ball_coords = ball_feats_thisbatch[:,:,-2:].to(torch.int64).cuda()  # the last 2 dims are [x, y], (B, T, 2)
            ball_coords_embeded = image_coords_learned[ball_coords[:, :, 1], ball_coords[:, :, 0]]  # (B, T, d)
            ball_feats_thisbatch = torch.cat([ball_feats_thisbatch[:,:,:-2],  # (B, T, 4)
                                              time_seq[:,0,0,:,:],  # (B, T, d)
                                              ball_coords_embeded], dim=-1)
            ball_track_feats_thisbatch_proj = self.ball_track_projection_layer(ball_feats_thisbatch.flatten(1,2)).unsqueeze(1)  # (B, 1, d)
            
            # Multiscale Transformer Blocks 
            outputs = self.TNT(CLS.transpose(0, 1),  # (1, B, d)
                               ball_track_feats_thisbatch_proj.transpose(0, 1),  # (1, B, d)
                               joint_track_feats_thisbatch_proj.transpose(0, 1),  # (N*J, B, d)
                               person_track_feats_thisbatch_proj.transpose(0, 1),  # (N, B, d)
                               interaction_track_feats_thisbatch_proj.transpose(0, 1),  # (N*(N-1), B, d)
                               group_track_feats_thisbatch_proj.transpose(0, 1),  # (2, B, d)
                               left_group_people_idx,
                               right_group_people_idx
                              )
        else:
            if self.args.dataset_name == 'volleyball':
                outputs = self.TNT(CLS.transpose(0, 1),  # (1, B, d)
                                   joint_track_feats_thisbatch_proj.transpose(0, 1),  # (N*J, B, d)
                                   person_track_feats_thisbatch_proj.transpose(0, 1),  # (N, B, d)
                                   interaction_track_feats_thisbatch_proj.transpose(0, 1),  # (N*(N-1), B, d)
                                   group_track_feats_thisbatch_proj.transpose(0, 1),  # (2, B, d)
                                   left_group_people_idx,
                                   right_group_people_idx
                                  )
            elif self.args.dataset_name == 'collective':
                outputs = self.TNT(CLS.transpose(0, 1),  # (1, B, d)
                                   joint_track_feats_thisbatch_proj.transpose(0, 1),  # (N*J, B, d)
                                   person_track_feats_thisbatch_proj.transpose(0, 1),  # (N, B, d)
                                   interaction_track_feats_thisbatch_proj.transpose(0, 1),  # (N*(N-1), B, d)
                                   group_track_feats_thisbatch_proj.transpose(0, 1)  # (2, B, d)
                                  )
            else:
                print('Please check the dataset name!')
                os._exit(0)
               
        # outputs is a list of list
        # len(outputs) is the numbr of TNT layers
        # each inner list is [CLS_f, CLS_m, CLS_c, output_CLS, output_fine, output_middle, output_coarse, output_group]
         
            
        # CLASSIFIER
        pred_logits = []
        for l in range(self.args.TNT_n_layers):
            
            fine_cls = outputs[l][0].transpose(0, 1).squeeze(1)  # (B, d)
            middle_cls = outputs[l][1].transpose(0, 1).squeeze(1)  # (B, d)
            coarse_cls = outputs[l][2].transpose(0, 1).squeeze(1)  # (B, d)
            group_cls = outputs[l][3].transpose(0, 1).squeeze(1)  # (B, d)
            
            pred_logit_f = self.classifier(fine_cls)
            pred_logit_m = self.classifier(middle_cls)
            pred_logit_c = self.classifier(coarse_cls)
            pred_logit_g = self.classifier(group_cls)
            
            pred_logits.append([pred_logit_f, pred_logit_m, pred_logit_c, pred_logit_g])
            
             
        # fine_cls, middle_cls, coarse_cls, group_cls are from the last layer
        fine_cls_normed = nn.functional.normalize(fine_cls, dim=1, p=2)
        middle_cls_normed = nn.functional.normalize(middle_cls, dim=1, p=2)
        coarse_cls_normed = nn.functional.normalize(coarse_cls, dim=1, p=2)
        group_cls_normed = nn.functional.normalize(group_cls, dim=1, p=2)

        scores_f = self.prototypes(fine_cls_normed)
        scores_m = self.prototypes(middle_cls_normed)
        scores_c = self.prototypes(coarse_cls_normed)
        scores_g = self.prototypes(group_cls_normed)
        scores = [scores_f, scores_m, scores_c, scores_g]
       
    
        # PERSON CLASSIFIER
        pred_logits_person = []
        for l in range(self.args.TNT_n_layers):
            person_feats = outputs[l][5].transpose(0, 1).flatten(0,1)  # (BxN, d)
            pred_logit_person = self.person_classifier(person_feats)  
            pred_logits_person.append(pred_logit_person)
        
        return pred_logits, pred_logits_person, scores
        
        
