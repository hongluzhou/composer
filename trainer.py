import os
import pdb
import numpy as np
import pickle
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from utils.common_utils import AverageMeter, accuracy



class Trainer():
    def __init__(self, args, model, logger, criterion, criterion_person, optimizer, 
                 train_loader=None, test_loader=None):
        
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.criterion = criterion 
        self.criterion_person = criterion_person
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        
    def train(self, epoch):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()  
        top1 = AverageMeter()
        top3 = AverageMeter()
        top1_person = AverageMeter()
        top3_person = AverageMeter()
        constras_loss = AverageMeter()  
        
        batch_start_time = time.time()
        
        self.model.train()
        self.criterion.train() 
        self.criterion_person.train()

        for i, batch_data in enumerate(self.train_loader):
            
            (joint_feats_thisbatch, targets_thisbatch, 
             video_thisbatch, clip_thisbatch, 
             person_labels, ball_feats) = batch_data
            
            data_time.update(time.time() - batch_start_time)
        
            self.optimizer.zero_grad()
            
            # normalize the prototypes
            with torch.no_grad():
                w = self.model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.module.prototypes.weight.copy_(w)
            
            # model forward pass
            pred_logits_thisbatch, pred_logits_person, scores = self.model(
                joint_feats_thisbatch, ball_feats)
                           
            
            # learning the cluster assignment and computing the loss
            scores_f = scores[0]
            scores_m = scores[1]
            scores_c = scores[2]
            scores_g = scores[3]

            # compute assignments
            with torch.no_grad(): 
                q_f = self.sinkhorn(scores_f, nmb_iters=self.args.sinkhorn_iterations)
                q_m = self.sinkhorn(scores_m, nmb_iters=self.args.sinkhorn_iterations)
                q_c = self.sinkhorn(scores_c, nmb_iters=self.args.sinkhorn_iterations)
                q_g = self.sinkhorn(scores_g, nmb_iters=self.args.sinkhorn_iterations)

            # swap prediction problem
            p_f = scores_f / self.args.temperature
            p_m = scores_m / self.args.temperature
            p_c = scores_c / self.args.temperature
            p_g = scores_g / self.args.temperature

            contrastive_clustering_loss = self.args.loss_coe_constrastive_clustering * (
                self.swap_prediction(p_f, p_m, q_f, q_m) + 
                self.swap_prediction(p_f, p_c, q_f, q_c) +
                self.swap_prediction(p_f, p_g, q_f, q_g) +
                self.swap_prediction(p_m, p_c, q_m, q_c) +
                self.swap_prediction(p_m, p_g, q_m, q_g) +
                self.swap_prediction(p_c, p_g, q_c, q_g)
            ) / 6.0  # 6 pairs of views

            constras_loss.update(contrastive_clustering_loss.data.item(), len(targets_thisbatch))
             
             
            # measure accuracy and record loss 
            targets_thisbatch = targets_thisbatch.to(pred_logits_thisbatch[0][0].device)
            person_labels = person_labels.flatten(0,1).to(pred_logits_thisbatch[0][0].device)
            
            loss_thisbatch, prec1, prec3, prec1_person, prec3_person = self.loss_acc_compute(
                pred_logits_thisbatch, targets_thisbatch, pred_logits_person, person_labels)
             
            loss_thisbatch += contrastive_clustering_loss
                    

            loss.update(loss_thisbatch.data.item(), len(targets_thisbatch))
            
            top1.update(prec1.item(), len(targets_thisbatch))
            top3.update(prec3.item(), len(targets_thisbatch))
            top1_person.update(prec1_person.item(), len(person_labels))
            top3_person.update(prec3_person.item(), len(person_labels))
    
            loss_thisbatch.backward()
            self.optimizer.step()
            
           
            # finish
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
              
            self.logger.info('Train [e{0:02d}][{1}/{2}] '
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f}) '
                  'Loss Constrastive {constras_loss.val:.4f}({constras_loss.avg:.4f}) '
                  'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                  'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                  'Person Top1 {top1_person.val:.4f}({top1_person.avg:.4f}) '
                  'Person Top3 {top3_person.val:.4f}({top3_person.avg:.4f}) '.format(
                      epoch, i+1, len(self.train_loader), batch_time=batch_time,
                      data_time=data_time, 
                      loss=loss, constras_loss=constras_loss, top1=top1, top3=top3,
                      top1_person=top1_person, top3_person=top3_person))
            
    
            
    @torch.no_grad()
    def test(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()  
        top1 = AverageMeter()
        top3 = AverageMeter()
        top1_person = AverageMeter()
        top3_person = AverageMeter()
        
        batch_start_time = time.time()
        
        self.model.eval()
        self.criterion.eval()
        self.criterion_person.eval()
        
        results = dict()

        for i, batch_data in enumerate(self.test_loader):
            
            (joint_feats_thisbatch, targets_thisbatch, 
             video_thisbatch, clip_thisbatch, 
             person_labels, ball_feats) = batch_data
           
            data_time.update(time.time() - batch_start_time)
            
            # normalize the prototypes
            with torch.no_grad():
                w = self.model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.module.prototypes.weight.copy_(w)
                    
            # model forward 
            pred_logits_thisbatch, pred_logits_person, _ = self.model(joint_feats_thisbatch, ball_feats)
            
            
            # measure accuracy and record loss 
            targets_thisbatch = targets_thisbatch.to(pred_logits_thisbatch[0][0].device)
            person_labels = person_labels.flatten(0,1).to(pred_logits_thisbatch[0][0].device)
            
            loss_thisbatch, prec1, prec3, prec1_person, prec3_person = self.loss_acc_compute(
                pred_logits_thisbatch, targets_thisbatch, pred_logits_person, person_labels)
            
            loss.update(loss_thisbatch.data.item(), len(targets_thisbatch))
            
            top1.update(prec1.item(), len(targets_thisbatch))
            top3.update(prec3.item(), len(targets_thisbatch))
            top1_person.update(prec1_person.item(), len(person_labels))
            top3_person.update(prec3_person.item(), len(person_labels))

            # update results
            predictions_this_batch = torch.argmax(pred_logits_thisbatch[-1][-1], dim=-1).cpu().numpy()
            predictions_this_batch_person = torch.argmax(pred_logits_person[-1], dim=-1).cpu().numpy()
                 
            for b_idx in range(len(video_thisbatch)):
                key = 'video_{}-clip_{}'.format(video_thisbatch[b_idx], clip_thisbatch[b_idx])
                assert key not in results
                results[key] = dict()
                results[key]['Pred'] = predictions_this_batch[b_idx]
                results[key]['GT'] = targets_thisbatch[b_idx].cpu().numpy()[()]
                results[key]['Pred_Person'] = []
                results[key]['GT_Person'] = []
                for p_idx in range(self.args.N):
                    results[key]['Pred_Person'].append(predictions_this_batch_person[b_idx*self.args.N+p_idx])
                    results[key]['GT_Person'].append(person_labels[b_idx*self.args.N+p_idx].cpu().numpy()[()])
                
           
            # finish
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            self.logger.info('Test [e{0:02d}][{1}/{2}] '
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f}) '
                  'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                  'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                  'Person Top1 {top1_person.val:.4f}({top1_person.avg:.4f}) '
                  'Person Top3 {top3_person.val:.4f}({top3_person.avg:.4f}) '.format(
                      epoch, i+1, len(self.test_loader), batch_time=batch_time,
                      data_time=data_time, 
                      loss=loss, top1=top1, top3=top3, 
                      top1_person=top1_person, top3_person=top3_person))
              
        return top1.avg, top3.avg, top1_person.avg, top3_person.avg, loss.avg, results
    
        
    def loss_acc_compute(
        self, pred_logits_thisbatch, targets_thisbatch, 
        pred_logits_person=None, person_labels=None):
        
        loss_thisbatch = 0
        
        for l in range(self.args.TNT_n_layers):
            if l == self.args.TNT_n_layers - 1:  # if last layer
                loss_thisbatch += self.args.loss_coe_last_TNT * (
                    self.args.loss_coe_fine * self.criterion(pred_logits_thisbatch[l][0], targets_thisbatch) + 
                    self.args.loss_coe_mid * self.criterion(pred_logits_thisbatch[l][1], targets_thisbatch) +
                    self.args.loss_coe_coarse * self.criterion(pred_logits_thisbatch[l][2], targets_thisbatch) +
                    self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][3], targets_thisbatch) +
                    self.args.loss_coe_person * self.criterion_person(pred_logits_person[l], person_labels))
                
                prec1, prec3 = accuracy(pred_logits_thisbatch[l][-1], targets_thisbatch, topk=(1, 3))
                prec1_person, prec3_person = accuracy(pred_logits_person[l], person_labels, topk=(1, 3))
                
            else:  # not last layer
                loss_thisbatch += (
                    self.args.loss_coe_fine * self.criterion(pred_logits_thisbatch[l][0], targets_thisbatch) + 
                    self.args.loss_coe_mid * self.criterion(pred_logits_thisbatch[l][1], targets_thisbatch) + 
                    self.args.loss_coe_coarse * self.criterion(pred_logits_thisbatch[l][2], targets_thisbatch) +
                    self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][3], targets_thisbatch) +
                    self.args.loss_coe_person * self.criterion_person(pred_logits_person[l], person_labels))
                
        return loss_thisbatch, prec1, prec3, prec1_person, prec3_person
        
        
    def sinkhorn(self, scores, epsilon=0.05, nmb_iters=3):
        with torch.no_grad():
            Q = torch.exp(scores / epsilon).t() 
            K, B = Q.shape
        
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            if len(self.args.gpu) > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
        
        
    def swap_prediction(self, p_t, p_s, q_t, q_s):
        loss = - 0.5 * (
            torch.mean(
                torch.sum(
                    q_t * F.log_softmax(p_s, dim=1), dim=1)
            ) + torch.mean(torch.sum(q_s * F.log_softmax(p_t, dim=1), dim=1)))
        return loss
        