import os
import pdb
import torch
import copy
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from models.transformer import TransformerEncoderLayer
from models.misc import build_mlp


class TNT(Module):
    def __init__(self, args, d_model, num_layers, final_norm=None, return_intermediate=False):
        '''
        d_model: representation dimension
        num_layers: number of TNT blocks
        final_norm: whether to use layer norm for final output
        return_intermediate: whether to return output from every TNT block
        '''
        super(TNT, self).__init__()
        
        encoder_layer = TNT_Block(args, d_model)
        self.layers = _get_clones(encoder_layer, num_layers)
        
        self.num_layers = num_layers
        self.final_norm = final_norm
        self.return_intermediate = return_intermediate

        if self.final_norm:
            self.norm_output_fine = LayerNorm(d_model)
            self.norm_output_middle = LayerNorm(d_model)
            self.norm_output_coarse = LayerNorm(d_model)
            self.norm_output_group =  LayerNorm(d_model)
            
            
    def forward(self, CLS, ball, fine, middle, coarse, group, left_group_people_idx, right_group_people_idx):
        '''
        CLS: (1, B, d)
        ball: (1, B, d)
        fine: (F, B, d)
        middle: (M, B, d)
        coarse: (C, B, d)
        group: (2, B, d)
        '''
        output_CLS = CLS
        output_ball = ball
        output_fine = fine
        output_middle = middle
        output_coarse = coarse
        output_group = group
        
        intermediate = []

        for mod in self.layers:
            CLS_f, CLS_m, CLS_c, output_CLS, output_ball, output_fine, output_middle, output_coarse, output_group = mod(
                output_CLS, output_ball, output_fine, output_middle, output_coarse, output_group, 
                left_group_people_idx, right_group_people_idx
            )
            if self.return_intermediate:
                intermediate.append(
                    [CLS_f, CLS_m, CLS_c, output_CLS, output_fine, output_middle, output_coarse, output_group])
                
        if self.final_norm is not None:
            CLS_f = self.norm_output_fine(CLS_f)
            CLS_m = self.norm_output_middle(CLS_m)
            CLS_c = self.norm_output_coarse(CLS_c)
            output_CLS = self.norm_output_group(output_CLS)
            output_fine = self.norm_output_fine(output_fine)
            output_middle = self.norm_output_middle(output_middle)
            output_coarse = self.norm_output_coarse(output_coarse)
            output_group = self.norm_output_group(output_group)
            
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append([CLS_f, CLS_m, CLS_c, output_CLS, output_fine, output_middle, output_coarse, output_group])
                
        if self.return_intermediate:
            return intermediate
        else:
            return CLS_f, CLS_m, CLS_c, output_CLS, output_fine, output_middle, output_coarse, output_group




class TNT_Block(Module):
    def __init__(self, args, d_model):
        super(TNT_Block, self).__init__()
        
        self.inner_tblock = TransformerEncoderLayer(
            d_model, args.innerTx_nhead, args.innerTx_dim_feedforward, 
            args.innerTx_dropout, args.innerTx_activation)
        
        self.inner2middle = Joint_To_Person(args, args.N, args.J, d_model)
        
        self.middle_tblock = TransformerEncoderLayer(
            d_model, args.middleTx_nhead, args.middleTx_dim_feedforward,
            args.middleTx_dropout, args.middleTx_activation)
        
        self.middle2coarse = Person_To_PersonInteraction(args, args.N, args.J, d_model)
        
        self.outer_tblock = TransformerEncoderLayer(
            d_model, args.outerTx_nhead, args.outerTx_dim_feedforward, 
            args.outerTx_dropout, args.outerTx_activation)
        
        self.middle2group = Person_To_Group(args.N//2, d_model)
        
        self.group_tblock = TransformerEncoderLayer(
            d_model, args.groupTx_nhead, args.groupTx_dim_feedforward,
            args.groupTx_dropout, args.outerTx_activation)
        
        
    def forward(self, CLS, ball, fine, middle, coarse, group, left_group_people_idx, right_group_people_idx):
        '''
        CLS: (1, B, d)
        fine: (F, B, d)
        middle: (M, B, d)
        coarse: (C, B, d)
        group: (2, B, d)
        '''
         
        output_inner = self.inner_tblock(torch.cat([CLS, ball, fine], dim=0))
        CLS_f = output_inner[0, :, :].unsqueeze(0)
        ball_f = output_inner[1, :, :].unsqueeze(0)
        output_fine = output_inner[2:, :, :]
        
        middle_update = middle + self.inner2middle(output_fine.transpose(0, 1)).transpose(0, 1)
        output_middle = self.middle_tblock(torch.cat([CLS_f, ball_f, middle_update], dim=0))
        CLS_m = output_middle[0, :, :].unsqueeze(0)
        ball_m = output_middle[1, :, :].unsqueeze(0)
        output_middle = output_middle[2:, :, :]
        
        coarse_update = coarse + self.middle2coarse(output_middle.transpose(0, 1)).transpose(0, 1)
        output_outer = self.outer_tblock(torch.cat([CLS_m, ball_m, coarse_update], dim=0))
        CLS_c = output_outer[0, :, :].unsqueeze(0)
        ball_c = output_outer[1, :, :].unsqueeze(0)
        output_coarse = output_outer[2:, :, :]
 
        group_update = group + self.middle2group(output_middle.transpose(0, 1), left_group_people_idx, right_group_people_idx).transpose(0, 1)
        output_group = self.group_tblock(torch.cat([CLS_c, ball_c, group_update], dim=0))
        CLS_g = output_group[0, :, :].unsqueeze(0)
        ball_g = output_group[1, :, :].unsqueeze(0)
        output_group = output_group[2:, :, :]
        
        return CLS_f, CLS_m, CLS_c, CLS_g, ball_g, output_fine, output_middle, output_coarse, output_group



class Joint_To_Person(Module):
    def __init__(self, args, N, J, d_model):
        super(Joint_To_Person, self).__init__()
        
        self.N, self.J, self.d = N, J, d_model
        
        # person identity projection layer
        self.person_projection_layer = build_mlp(input_dim=J*d_model,
                                                 hidden_dims=[d_model], 
                                                 output_dim=d_model)
                      
    def forward(self, joint_feats_thisbatch):
        # joint_feats_thisbatch: (B, N*J, d)
        B = joint_feats_thisbatch.size(0)
        N, J, d = self.N, self.J, self.d
        
        # person identity projection
        person_feats_thisbatch_proj = self.person_projection_layer(
            joint_feats_thisbatch.view(-1, N, J, d).flatten(2, 3)  # (B, N, J*d)
        )
        # (B, N, d)
        return person_feats_thisbatch_proj
    
    
    
class Person_To_PersonInteraction(Module):
    def __init__(self, args, N, J, d_model):
        super(Person_To_PersonInteraction, self).__init__()
        
        self.N, self.J, self.d = N, J, d_model
        
        self.interaction_indexes = [N*i+j for i in range(N) for j in range(N) if N*i+j != N*i+i]
        
        # person interaction projection layer
        self.interaction_projection_layer = build_mlp(input_dim=d_model*2,
                                                      hidden_dims=[d_model], 
                                                      output_dim=d_model)
                      
    def forward(self, person_feats_thisbatch_proj):
        B = person_feats_thisbatch_proj.size(0)
        N, J, d = self.N, self.J, self.d
        
        # form sequence of person-person-interaction tokens
        tem1 = person_feats_thisbatch_proj.repeat(1, N, 1).reshape(B,N,N,d).transpose(1, 2).flatten(1, 2)  # (B, N^2, d)
        tem2 = person_feats_thisbatch_proj.repeat(1, N, 1) # (B, N^2, d)
        tem3 = torch.cat([tem1, tem2], dim=-1)  # (B, N^2, 2*d)
        interactions_thisbatch = tem3[:, self.interaction_indexes, :]  # (B, N*(N-1), 2*d)
        interactions_thisbatch_proj = self.interaction_projection_layer(interactions_thisbatch)  # (B, N*(N-1), d)
        return interactions_thisbatch_proj


class Person_To_Group(Module):
    def __init__(self, num_person_per_group, d_model):
        super(Person_To_Group, self).__init__()
        self.num_person_per_group = num_person_per_group
        self.person_to_group_projection = build_mlp(input_dim=num_person_per_group*d_model,
                                                      hidden_dims=[d_model], 
                                                      output_dim=d_model)
                      
    def forward(self, person_feats_thisbatch_proj, left_group_people_idx, right_group_people_idx):
        B = person_feats_thisbatch_proj.size(0)
        # form sequence of group track tokens
        left_group_people_repre = person_feats_thisbatch_proj.flatten(
            0,1)[left_group_people_idx.flatten(0,1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        right_group_people_repre = person_feats_thisbatch_proj.flatten(
            0,1)[right_group_people_idx.flatten(0,1)].view(B, self.num_person_per_group, -1)  # (B, N/2, d)
        left_group_feats_thisbatch_proj = self.person_to_group_projection(left_group_people_repre.flatten(1,2))   # (B, d)
        right_group_feats_thisbatch_proj = self.person_to_group_projection(right_group_people_repre.flatten(1,2))   # (B, d)
        group_track_feats_thisbatch_proj = torch.stack([left_group_feats_thisbatch_proj, right_group_feats_thisbatch_proj], dim=1)  # (B, 2, d)

        return group_track_feats_thisbatch_proj
    

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
