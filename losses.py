import numpy as np
import torch
import torch.nn as nn
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
from tools.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from tools.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, gaussian_radius_1
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, TripletLoss, RegLoss_weighted, RegLoss_whv
import models.losses as ls
from models.utils import _sigmoid, _tranpose_and_gather_feat
import math
import copy


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class TotalLoss_vis_atten_thm_4(nn.Module):
    def __init__(self):
        super(TotalLoss_vis_atten_thm_4, self).__init__()
        self.crit = ls.FocalLoss()
        # self.crit = ls.FocalLoss_vis()
        self.crit_reid = self.crit
        self.crit_reg = RegLoss()
        self.crit_vis = RegLoss_weighted()
        self.emb_dim = 256
        self.nID = 537
        # self.nID = 2209
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, classifications, regressions, viss, identis, annotations1, annotations2, s_det, s_id):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        vis_losses = []
        id_losses = []
        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            vis = viss[j, :, :]
            # offset = offsets[j, :, :]
            identi = identis[j, :, :]
            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            classification = torch.sigmoid(classification.clone())
            classification = torch.clamp(classification, min=1e-4, max=1 - 1e-4)
            vis = torch.sigmoid(vis.clone())
            vis = torch.clamp(vis, min=1e-4, max=1 - 1e-4)
            # print(reid.shape)
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                vis_losses.append(torch.tensor(0).float().cuda())
                id_losses.append(torch.tensor(0).float().cuda())
                continue

            output_h = 608 // 4  # 输出分辨率
            output_w = 1088 // 4
            max_objs = 256
            num_classes = 1
            # print(bbox_annotation.shape)
            num_objs = bbox_annotation.shape[0]
            # print(num_objs)
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float16)
            wh = np.zeros((max_objs, 8), dtype=np.float16)  # # left right top bottom
            v = np.zeros((max_objs,), dtype=np.float16)
            ind = np.zeros((max_objs,), dtype=np.int64)
            reg_mask = np.zeros((max_objs,), dtype=np.uint8)
            ids = np.zeros((max_objs,), dtype=np.int64)
            draw_gaussian = draw_umich_gaussian
            for k in range(num_objs):
                # print(annot[k])
                label = bbox_annotation[k]  # 8 dim
                bbox = label[0:4]
                cls_id = int(label[-4])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h

                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0].cpu(), 0, output_w - 1)
                bbox[1] = np.clip(bbox[1].cpu(), 0, output_h - 1)

                h = bbox[3]
                w = bbox[2]

                p_id = label[-3]
                flag = 0

                radiu0 = gaussian_radius((math.ceil(h), math.ceil(w)))
                radiu0_vis = radiu0 * torch.exp(label[-1]-1)
                radius = max(0, int(radiu0))
                # print(radiu0_vis.round())
                radius_vis = max(1, int(radiu0_vis))

                # print("{0}  {1} {2} {3}".format(radiu0, radiu0_vis, int(radiu0), int(radiu0_vis.round())))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float16)
                ct_int = ct.astype(np.int32)
                # if label[-1] < 0:
                #     print(bbox_amodal, w, h, ct_int)
                draw_gaussian(hm[cls_id], ct_int, radius)
                # draw_gaussian(hm[cls_id], ct_int, radius_vis)
                for t1 in bbox_annotation_next:
                    if p_id == t1[-3]:
                        flag = 1
                        bbox_t1 = t1[0:4]
                        bbox_t1[[0, 2]] = bbox_t1[[0, 2]] * output_w
                        bbox_t1[[1, 3]] = bbox_t1[[1, 3]] * output_h

                        bbox_amodal_t1 = copy.deepcopy(bbox_t1)

                        # get precise xyxy
                        bbox_amodal_t1[0] = bbox_amodal_t1[0] - bbox_amodal_t1[2] / 2.
                        bbox_amodal_t1[1] = bbox_amodal_t1[1] - bbox_amodal_t1[3] / 2.
                        bbox_amodal_t1[2] = bbox_amodal_t1[0] + bbox_amodal_t1[2]
                        bbox_amodal_t1[3] = bbox_amodal_t1[1] + bbox_amodal_t1[3]

                        bbox_t1[0] = np.clip(bbox_t1[0].cpu(), 0, output_w - 1)
                        bbox_t1[1] = np.clip(bbox_t1[1].cpu(), 0, output_h - 1)
                        h_t1 = bbox_t1[3]
                        w_t1 = bbox_t1[2]
                        mid_p = (bbox[:2] + bbox_t1[:2]) / 2
                        ct_t1 = np.array([bbox_t1[0], bbox_t1[1]], dtype=np.float16)
                        # print((h/2)+ct[1])
                        # if bbox_amodal[3]<0:
                        # 
                        #     print(w, h, w_t1, h_t1)
                        if (h > 0 and w > 0) and (h_t1 > 0 and w_t1 > 0):
                            # if (ct_int[0] - ct1_int[0]) == 0 and (ct_int[1] - ct1_int[1]) == 0:

                            ind[k] = ct_int[1].copy() * output_w + ct_int[0].copy()
                            # wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], bbox_amodal[2] - ct[0], \
                            #         bbox_amodal[3] - ct[1], ct[0] - bbox_amodal_t1[0], ct[1] - bbox_amodal_t1[1], \
                            #         bbox_amodal_t1[2] - ct[0], bbox_amodal_t1[3] - ct[1]
                            # reg[k] = (ct - ct_int)[0], (ct - ct_int)[1], (ct_t1 - ct_int)[0], (ct_t1 - ct_int)[1]

                            # left right top bottom
                            wh[k] = ct_int[0] - bbox_amodal[0], bbox_amodal[2] - ct_int[0], ct_int[1] - bbox_amodal[
                                1], \
                                    bbox_amodal[3] - ct_int[1], \
                                    ct_int[0] - bbox_amodal_t1[0], bbox_amodal_t1[2] - ct_int[0], ct_int[1] - \
                                    bbox_amodal_t1[1], \
                                    bbox_amodal_t1[3] - ct_int[1]
                            reg_mask[k] = 1
                            ids[k] = p_id
                            v[k] = label[-2]
                            # ids[k] = label[-3]
                            break
            classification = classification.reshape(1, 1, output_h, output_w)
            hm = torch.from_numpy(hm.reshape((1, 1, output_h, output_w))).cuda(non_blocking=True)

            regression = torch.reshape(regression, (1, 8, output_h, output_w))
            vis = torch.reshape(vis, (1, 1, output_h, output_w))
            identi = torch.reshape(identi, (1, self.emb_dim, output_h, output_w))
            ind = torch.from_numpy(ind.reshape(1, max_objs)).cuda(non_blocking=True)
            reg_mask = torch.from_numpy(reg_mask.reshape(1, max_objs)).cuda(non_blocking=True)
            wh = torch.from_numpy(wh.reshape((1, max_objs, 8))).cuda(non_blocking=True)
            v = torch.from_numpy(v.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            ids = torch.from_numpy(ids.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            # print(regression)

            reg_loss = self.crit_reg(
                regression, reg_mask,
                ind, wh)
            vis_loss = self.crit_vis(
                vis, reg_mask,
                ind, v)
            # print(reg)

            id_head = _tranpose_and_gather_feat(identi, ind)
            id_head = id_head[reg_mask > 0].contiguous()
            id_head = self.emb_scale * torch.nn.functional.normalize(id_head)
            id_target = ids[reg_mask > 0]
            id_weight = v[reg_mask > 0]
            id_output = self.classifier(id_head).contiguous()

            one_hot = torch.nn.functional.one_hot(id_target, 537).reshape(-1,537).float()
            softmax = torch.exp(id_output) / torch.sum(torch.exp(id_output), dim=1).reshape(-1, 1)

            logsoftmax = torch.log(softmax)

            # print(id_weight.shape)
            # print(one_hot.shape)
            # print(logsoftmax.shape)
            
            nllloss = -torch.sum(id_weight * one_hot * logsoftmax) / id_target.shape[0]

            # id_losses.append(self.IDLoss(id_output, id_target.squeeze(1)))
            id_losses.append(nllloss)

            vis_losses.append(vis_loss.half())
            classification_losses.append(self.crit(classification, hm).half())
            regression_losses.append(reg_loss.half())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0,
            keepdim=True), torch.stack(
            vis_losses).mean(dim=0, keepdim=True), torch.stack(
            id_losses).mean(dim=0, keepdim=True),s_det, s_id


class TotalLoss_vis_atten_thm_4_vis(nn.Module):
    def __init__(self):
        super(TotalLoss_vis_atten_thm_4_vis, self).__init__()
        self.crit = ls.FocalLoss()
        # self.crit = ls.FocalLoss_vis()
        self.crit_reid = self.crit
        self.crit_reg = RegLoss_whv()
        self.crit_vis = RegLoss_weighted()
        self.emb_dim = 256
        self.nID = 537
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, classifications, regressions, viss, identis, annotations1, annotations2, s_det, s_id):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        vis_losses = []
        id_losses = []
        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            vis = viss[j, :, :]
            # offset = offsets[j, :, :]
            identi = identis[j, :, :]
            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            classification = torch.sigmoid(classification.clone())
            classification = torch.clamp(classification, min=1e-4, max=1 - 1e-4)
            vis = torch.sigmoid(vis.clone())
            vis = torch.clamp(vis, min=1e-4, max=1 - 1e-4)
            # print(reid.shape)
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                vis_losses.append(torch.tensor(0).float().cuda())
                id_losses.append(torch.tensor(0).float().cuda())
                continue

            output_h = 608 // 4  # 输出分辨率
            output_w = 1088 // 4
            max_objs = 128
            num_classes = 1
            # print(bbox_annotation.shape)
            num_objs = bbox_annotation.shape[0]
            # print(num_objs)
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float16)
            wh = np.zeros((max_objs, 8), dtype=np.float16)  # # left right top bottom
            v = np.zeros((max_objs,), dtype=np.float16)
            ind = np.zeros((max_objs,), dtype=np.int64)
            reg_mask = np.zeros((max_objs,), dtype=np.uint8)
            ids = np.zeros((max_objs,), dtype=np.int64)
            draw_gaussian = draw_umich_gaussian
            for k in range(num_objs):
                # print(annot[k])
                label = bbox_annotation[k]  # 8 dim
                bbox = label[0:4]
                cls_id = int(label[-4])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h

                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0].cpu(), 0, output_w - 1)
                bbox[1] = np.clip(bbox[1].cpu(), 0, output_h - 1)

                h = bbox[3]
                w = bbox[2]

                p_id = label[-3]
                flag = 0

                radiu0 = gaussian_radius((math.ceil(h), math.ceil(w)))
                radiu0_vis = radiu0 * torch.exp(label[-1] - 1)
                radius = max(0, int(radiu0))
                # print(radiu0_vis.round())
                radius_vis = max(1, int(radiu0_vis))

                # print("{0}  {1} {2} {3}".format(radiu0, radiu0_vis, int(radiu0), int(radiu0_vis.round())))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float16)
                ct_int = ct.astype(np.int32)
                # if label[-1] < 0:
                #     print(bbox_amodal, w, h, ct_int)
                draw_gaussian(hm[cls_id], ct_int, radius)
                # draw_gaussian(hm[cls_id], ct_int, radius_vis)
                for t1 in bbox_annotation_next:
                    if p_id == t1[-3]:
                        flag = 1
                        bbox_t1 = t1[0:4]
                        bbox_t1[[0, 2]] = bbox_t1[[0, 2]] * output_w
                        bbox_t1[[1, 3]] = bbox_t1[[1, 3]] * output_h

                        bbox_amodal_t1 = copy.deepcopy(bbox_t1)

                        # get precise xyxy
                        bbox_amodal_t1[0] = bbox_amodal_t1[0] - bbox_amodal_t1[2] / 2.
                        bbox_amodal_t1[1] = bbox_amodal_t1[1] - bbox_amodal_t1[3] / 2.
                        bbox_amodal_t1[2] = bbox_amodal_t1[0] + bbox_amodal_t1[2]
                        bbox_amodal_t1[3] = bbox_amodal_t1[1] + bbox_amodal_t1[3]

                        bbox_t1[0] = np.clip(bbox_t1[0].cpu(), 0, output_w - 1)
                        bbox_t1[1] = np.clip(bbox_t1[1].cpu(), 0, output_h - 1)
                        h_t1 = bbox_t1[3]
                        w_t1 = bbox_t1[2]
                        mid_p = (bbox[:2] + bbox_t1[:2]) / 2
                        ct_t1 = np.array([bbox_t1[0], bbox_t1[1]], dtype=np.float16)
                        # print((h/2)+ct[1])
                        # if bbox_amodal[3]<0:
                        #
                        #     print(w, h, w_t1, h_t1)
                        if (h > 0 and w > 0) and (h_t1 > 0 and w_t1 > 0):
                            # if (ct_int[0] - ct1_int[0]) == 0 and (ct_int[1] - ct1_int[1]) == 0:

                            ind[k] = ct_int[1].copy() * output_w + ct_int[0].copy()
                            # wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], bbox_amodal[2] - ct[0], \
                            #         bbox_amodal[3] - ct[1], ct[0] - bbox_amodal_t1[0], ct[1] - bbox_amodal_t1[1], \
                            #         bbox_amodal_t1[2] - ct[0], bbox_amodal_t1[3] - ct[1]
                            # reg[k] = (ct - ct_int)[0], (ct - ct_int)[1], (ct_t1 - ct_int)[0], (ct_t1 - ct_int)[1]

                            # left right top bottom
                            wh[k] = ct_int[0] - bbox_amodal[0], bbox_amodal[2] - ct_int[0], ct_int[1] - bbox_amodal[
                                1], \
                                    bbox_amodal[3] - ct_int[1], \
                                    ct_int[0] - bbox_amodal_t1[0], bbox_amodal_t1[2] - ct_int[0], ct_int[1] - \
                                    bbox_amodal_t1[1], \
                                    bbox_amodal_t1[3] - ct_int[1]
                            reg_mask[k] = 1
                            ids[k] = p_id
                            v[k] = label[-2]
                            # ids[k] = label[-3]
                            break
            classification = classification.reshape(1, 1, output_h, output_w)
            hm = torch.from_numpy(hm.reshape((1, 1, output_h, output_w))).cuda(non_blocking=True)

            regression = torch.reshape(regression, (1, 8, output_h, output_w))
            vis = torch.reshape(vis, (1, 1, output_h, output_w))
            identi = torch.reshape(identi, (1, self.emb_dim, output_h, output_w))
            ind = torch.from_numpy(ind.reshape(1, max_objs)).cuda(non_blocking=True)
            reg_mask = torch.from_numpy(reg_mask.reshape(1, max_objs)).cuda(non_blocking=True)
            wh = torch.from_numpy(wh.reshape((1, max_objs, 8))).cuda(non_blocking=True)
            v = torch.from_numpy(v.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            ids = torch.from_numpy(ids.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            # print(regression)

            reg_loss = self.crit_reg(
                regression, reg_mask,
                ind, wh, v)
            vis_loss = self.crit_vis(
                vis, reg_mask,
                ind, v)
            # print(reg)

            id_head = _tranpose_and_gather_feat(identi, ind)
            id_head = id_head[reg_mask > 0].contiguous()
            id_head = self.emb_scale * torch.nn.functional.normalize(id_head)
            id_target = ids[reg_mask > 0]
            id_weight = v[reg_mask > 0]
            id_output = self.classifier(id_head).contiguous()

            one_hot = torch.nn.functional.one_hot(id_target, 537).reshape(-1, 537).float()
            softmax = torch.exp(id_output) / torch.sum(torch.exp(id_output), dim=1).reshape(-1, 1)

            logsoftmax = torch.log(softmax)

            # print(id_weight.shape)
            # print(one_hot.shape)
            # print(logsoftmax.shape)

            nllloss = -torch.sum(id_weight * one_hot * logsoftmax) / id_target.shape[0]

            # id_losses.append(self.IDLoss(id_output, id_target.squeeze(1)))
            id_losses.append(nllloss)

            vis_losses.append(vis_loss.half())
            classification_losses.append(self.crit(classification, hm).half())
            regression_losses.append(reg_loss.half())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0,
            keepdim=True), torch.stack(
            vis_losses).mean(dim=0, keepdim=True), torch.stack(
            id_losses).mean(dim=0, keepdim=True), s_det, s_id
class TotalLoss_vis_atten_thm_fair_4(nn.Module):
    def __init__(self):
        super(TotalLoss_vis_atten_thm_fair_4, self).__init__()
        self.crit = ls.FocalLoss()
        self.crit_reid = self.crit
        self.crit_reg = RegLoss()
        self.crit_vis = RegLoss_weighted()
        self.emb_dim = 256
        self.nID = 537
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, classifications, regressions, offsets, viss, identis, annotations1, annotations2, s_det, s_id):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        offset_losses = []
        vis_losses = []
        id_losses = []
        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            vis = viss[j, :, :]
            offset = offsets[j, :, :]
            identi = identis[j, :, :]
            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            classification = torch.sigmoid(classification.clone())
            classification = torch.clamp(classification, min=1e-4, max=1 - 1e-4)
            vis = torch.sigmoid(vis.clone())
            vis = torch.clamp(vis, min=1e-4, max=1 - 1e-4)
            # print(reid.shape)
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                vis_losses.append(torch.tensor(0).float().cuda())
                id_losses.append(torch.tensor(0).float().cuda())
                offset_losses.append(torch.tensor(0).float().cuda())
                continue

            output_h = 608 // 4  # 输出分辨率
            output_w = 1088 // 4
            max_objs = 128
            num_classes = 1
            # print(bbox_annotation.shape)
            num_objs = bbox_annotation.shape[0]
            # print(num_objs)
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float16)
            wh = np.zeros((max_objs, 8), dtype=np.float16)  # # left right top bottom
            reg = np.zeros((max_objs, 4), dtype=np.float16)
            v = np.zeros((max_objs,), dtype=np.float16)
            ind = np.zeros((max_objs,), dtype=np.int64)
            reg_mask = np.zeros((max_objs,), dtype=np.uint8)
            ids = np.zeros((max_objs,), dtype=np.int64)
            draw_gaussian = draw_umich_gaussian
            for k in range(num_objs):
                # print(annot[k])
                label = bbox_annotation[k]
                bbox = label[0:4]
                cls_id = int(label[-3])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h

                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0].cpu(), 0, output_w - 1)
                bbox[1] = np.clip(bbox[1].cpu(), 0, output_h - 1)

                h = bbox[3]
                w = bbox[2]

                p_id = label[-2]
                flag = 0

                radiu0 = gaussian_radius((math.ceil(h), math.ceil(w)))
                radiu0_vis = radiu0 * torch.exp(label[-1] - 1)
                radius = max(0, int(radiu0))
                # print(radiu0_vis.round())
                radius_vis = max(0, int(radiu0_vis))

                # print("{0}  {1} {2} {3}".format(radiu0, radiu0_vis, int(radiu0), int(radiu0_vis.round())))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float16)
                ct_int = ct.astype(np.int32)

                for t1 in bbox_annotation_next:
                    if p_id == t1[-2]:
                        flag = 1
                        bbox_t1 = t1[0:4]
                        bbox_t1[[0, 2]] = bbox_t1[[0, 2]] * output_w
                        bbox_t1[[1, 3]] = bbox_t1[[1, 3]] * output_h

                        bbox_amodal_t1 = copy.deepcopy(bbox_t1)

                        # get precise xyxy
                        bbox_amodal_t1[0] = bbox_amodal_t1[0] - bbox_amodal_t1[2] / 2.
                        bbox_amodal_t1[1] = bbox_amodal_t1[1] - bbox_amodal_t1[3] / 2.
                        bbox_amodal_t1[2] = bbox_amodal_t1[0] + bbox_amodal_t1[2]
                        bbox_amodal_t1[3] = bbox_amodal_t1[1] + bbox_amodal_t1[3]

                        bbox_t1[0] = np.clip(bbox_t1[0].cpu(), 0, output_w - 1)
                        bbox_t1[1] = np.clip(bbox_t1[1].cpu(), 0, output_h - 1)
                        h_t1 = bbox_t1[3]
                        w_t1 = bbox_t1[2]
                        mid_p = (bbox[:2] + bbox_t1[:2]) / 2
                        ct_t1 = np.array([bbox_t1[0], bbox_t1[1]], dtype=np.float16)
                        if (h > 0 and w > 0) and (h_t1 > 0 and w_t1 > 0):
                            # if (ct_int[0] - ct1_int[0]) == 0 and (ct_int[1] - ct1_int[1]) == 0:
                            draw_gaussian(hm[cls_id], ct_int, radius_vis)
                            ind[k] = ct_int[1].copy() * output_w + ct_int[0].copy()
                            wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], bbox_amodal[2] - ct[0], \
                                    bbox_amodal[3] - ct[1], ct[0] - bbox_amodal_t1[0], ct[1] - bbox_amodal_t1[1], \
                                    bbox_amodal_t1[2] - ct[0], bbox_amodal_t1[3] - ct[1]
                            reg[k] = (ct - ct_int)[0], (ct - ct_int)[1], (ct_t1 - ct_int)[0], (ct_t1 - ct_int)[1]

                            # left right top bottom
                            # wh[k] = ct_int[0] - bbox_amodal[0], bbox_amodal[2] - ct_int[0], ct_int[1] - bbox_amodal[
                            #     1], \
                            #         bbox_amodal[3] - ct_int[1], \
                            #         ct_int[0] - bbox_amodal_t1[0], bbox_amodal_t1[2] - ct_int[0], ct_int[1] - \
                            #         bbox_amodal_t1[1], \
                            #         bbox_amodal_t1[3] - ct_int[1]
                            reg_mask[k] = 1
                            ids[k] = p_id
                            v[k] = label[-1]
                            ids[k] = label[-2]
                            break
            classification = classification.reshape(1, 1, output_h, output_w)
            hm = torch.from_numpy(hm.reshape((1, 1, output_h, output_w))).cuda(non_blocking=True)

            regression = torch.reshape(regression, (1, 8, output_h, output_w))
            offset = torch.reshape(offset, (1, 4, output_h, output_w))
            vis = torch.reshape(vis, (1, 1, output_h, output_w))
            identi = torch.reshape(identi, (1, self.emb_dim, output_h, output_w))
            ind = torch.from_numpy(ind.reshape(1, max_objs)).cuda(non_blocking=True)
            reg_mask = torch.from_numpy(reg_mask.reshape(1, max_objs)).cuda(non_blocking=True)
            wh = torch.from_numpy(wh.reshape((1, max_objs, 8))).cuda(non_blocking=True)
            reg = torch.from_numpy(reg.reshape((1, max_objs, 4))).cuda(non_blocking=True)
            v = torch.from_numpy(v.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            ids = torch.from_numpy(ids.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            # print(regression)

            reg_loss = self.crit_reg(
                regression, reg_mask,
                ind, wh)
            off_loss = self.crit_reg(
                offset, reg_mask,
                ind, reg)
            vis_loss = self.crit_vis(
                vis, reg_mask,
                ind, v)
            # print(reg)

            id_head = _tranpose_and_gather_feat(identi, ind)
            id_head = id_head[reg_mask > 0].contiguous()
            id_head = self.emb_scale * torch.nn.functional.normalize(id_head)
            id_target = ids[reg_mask > 0]
            id_weight = v[reg_mask > 0]
            id_output = self.classifier(id_head).contiguous()

            one_hot = torch.nn.functional.one_hot(id_target, 537).reshape(-1, 537).float()
            softmax = torch.exp(id_output) / torch.sum(torch.exp(id_output), dim=1).reshape(-1, 1)

            logsoftmax = torch.log(softmax)

            # print(id_weight.shape)
            # print(one_hot.shape)
            # print(logsoftmax.shape)

            nllloss = -torch.sum(id_weight * one_hot * logsoftmax) / id_target.shape[0]

            # id_losses.append(self.IDLoss(id_output, id_target.squeeze(1)))
            id_losses.append(nllloss.half())
            offset_losses.append(off_loss.half())
            vis_losses.append(vis_loss.half())
            classification_losses.append(self.crit(classification, hm).half())
            regression_losses.append(reg_loss.half())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0,
            keepdim=True), torch.stack(offset_losses).mean(
            dim=0,
            keepdim=True),torch.stack(
            vis_losses).mean(dim=0, keepdim=True), torch.stack(
            id_losses).mean(dim=0, keepdim=True), s_det, s_id


class TotalLoss_vis_atten_thm_5(nn.Module):
    def __init__(self):
        super(TotalLoss_vis_atten_thm_5, self).__init__()
        self.crit = ls.FocalLoss()
        # self.crit = ls.FocalLoss_vis()
        self.crit_reid = self.crit
        self.crit_reg = RegLoss()
        self.crit_vis = RegLoss_weighted()
        self.emb_dim_64 = 64
        self.emb_dim_128 = 128
        
        self.emb_dim_256 = 256
        self.emb_dim_end = 128
        self.nID = 537
        self.classifier_64 = nn.Linear(self.emb_dim_64, self.nID)
        self.classifier_128 = nn.Linear(self.emb_dim_128, self.nID)
        self.classifier_256 = nn.Linear(self.emb_dim_256, self.nID)
        self.classifier_end = nn.Linear(self.emb_dim_end, self.nID)

        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, classifications, regressions, viss, identis_64, identis_128, identis_256, identis_end, annotations1, annotations2, s_det, s_id):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        vis_losses = []
        id_losses = []
        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            vis = viss[j, :, :]
            # offset = offsets[j, :, :]
            identi_64 = identis_64[j, :, :]
            identi_128 = identis_128[j, :, :]
            identi_256 = identis_256[j, :, :]
            identi_end= identis_end[j, :, :]
            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            classification = torch.sigmoid(classification.clone())
            classification = torch.clamp(classification, min=1e-4, max=1 - 1e-4)
            vis = torch.sigmoid(vis.clone())
            vis = torch.clamp(vis, min=1e-4, max=1 - 1e-4)
            # print(reid.shape)
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                vis_losses.append(torch.tensor(0).float().cuda())
                id_losses.append(torch.tensor(0).float().cuda())
                continue

            output_h = 608 // 4  # 输出分辨率
            output_w = 1088 // 4
            max_objs = 128
            num_classes = 1
            # print(bbox_annotation.shape)
            num_objs = bbox_annotation.shape[0]
            # print(num_objs)
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float16)
            wh = np.zeros((max_objs, 8), dtype=np.float16)  # # left right top bottom
            v = np.zeros((max_objs,), dtype=np.float16)
            ind = np.zeros((max_objs,), dtype=np.int64)
            reg_mask = np.zeros((max_objs,), dtype=np.uint8)
            ids = np.zeros((max_objs,), dtype=np.int64)
            draw_gaussian = draw_umich_gaussian
            for k in range(num_objs):
                # print(annot[k])
                label = bbox_annotation[k]  # 8 dim
                bbox = label[0:4]
                cls_id = int(label[-4])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h

                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0].cpu(), 0, output_w - 1)
                bbox[1] = np.clip(bbox[1].cpu(), 0, output_h - 1)

                h = bbox[3]
                w = bbox[2]

                p_id = label[-3]
                flag = 0

                radiu0 = gaussian_radius((math.ceil(h), math.ceil(w)))
                radiu0_vis = radiu0 * torch.exp(label[-1] - 1)
                radius = max(0, int(radiu0))
                # print(radiu0_vis.round())
                radius_vis = max(1, int(radiu0_vis))

                # print("{0}  {1} {2} {3}".format(radiu0, radiu0_vis, int(radiu0), int(radiu0_vis.round())))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float16)
                ct_int = ct.astype(np.int32)
                # if label[-1] < 0:
                #     print(bbox_amodal, w, h, ct_int)
                draw_gaussian(hm[cls_id], ct_int, radius)
                # draw_gaussian(hm[cls_id], ct_int, radius_vis)
                for t1 in bbox_annotation_next:
                    if p_id == t1[-3]:
                        flag = 1
                        bbox_t1 = t1[0:4]
                        bbox_t1[[0, 2]] = bbox_t1[[0, 2]] * output_w
                        bbox_t1[[1, 3]] = bbox_t1[[1, 3]] * output_h

                        bbox_amodal_t1 = copy.deepcopy(bbox_t1)

                        # get precise xyxy
                        bbox_amodal_t1[0] = bbox_amodal_t1[0] - bbox_amodal_t1[2] / 2.
                        bbox_amodal_t1[1] = bbox_amodal_t1[1] - bbox_amodal_t1[3] / 2.
                        bbox_amodal_t1[2] = bbox_amodal_t1[0] + bbox_amodal_t1[2]
                        bbox_amodal_t1[3] = bbox_amodal_t1[1] + bbox_amodal_t1[3]

                        bbox_t1[0] = np.clip(bbox_t1[0].cpu(), 0, output_w - 1)
                        bbox_t1[1] = np.clip(bbox_t1[1].cpu(), 0, output_h - 1)
                        h_t1 = bbox_t1[3]
                        w_t1 = bbox_t1[2]
                        mid_p = (bbox[:2] + bbox_t1[:2]) / 2
                        ct_t1 = np.array([bbox_t1[0], bbox_t1[1]], dtype=np.float16)
                        # print((h/2)+ct[1])
                        # if bbox_amodal[3]<0:
                        # 
                        #     print(w, h, w_t1, h_t1)
                        if (h > 0 and w > 0) and (h_t1 > 0 and w_t1 > 0):
                            # if (ct_int[0] - ct1_int[0]) == 0 and (ct_int[1] - ct1_int[1]) == 0:

                            ind[k] = ct_int[1].copy() * output_w + ct_int[0].copy()
                            # wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], bbox_amodal[2] - ct[0], \
                            #         bbox_amodal[3] - ct[1], ct[0] - bbox_amodal_t1[0], ct[1] - bbox_amodal_t1[1], \
                            #         bbox_amodal_t1[2] - ct[0], bbox_amodal_t1[3] - ct[1]
                            # reg[k] = (ct - ct_int)[0], (ct - ct_int)[1], (ct_t1 - ct_int)[0], (ct_t1 - ct_int)[1]

                            # left right top bottom
                            wh[k] = ct_int[0] - bbox_amodal[0], bbox_amodal[2] - ct_int[0], ct_int[1] - bbox_amodal[
                                1], \
                                    bbox_amodal[3] - ct_int[1], \
                                    ct_int[0] - bbox_amodal_t1[0], bbox_amodal_t1[2] - ct_int[0], ct_int[1] - \
                                    bbox_amodal_t1[1], \
                                    bbox_amodal_t1[3] - ct_int[1]
                            reg_mask[k] = 1
                            ids[k] = p_id
                            v[k] = label[-2]
                            # ids[k] = label[-3]
                            break
            classification = classification.reshape(1, 1, output_h, output_w)
            hm = torch.from_numpy(hm.reshape((1, 1, output_h, output_w))).cuda(non_blocking=True)

            regression = torch.reshape(regression, (1, 8, output_h, output_w))
            vis = torch.reshape(vis, (1, 1, output_h, output_w))
            identi_64 = torch.reshape(identi_64, (1, self.emb_dim_64, output_h, output_w))
            identi_128 = torch.reshape(identi_128, (1, self.emb_dim_128, output_h, output_w))
            identi_256 = torch.reshape(identi_256, (1, self.emb_dim_256, output_h, output_w))
            identi_end = torch.reshape(identi_end, (1, self.emb_dim_end, output_h, output_w))
            ind = torch.from_numpy(ind.reshape(1, max_objs)).cuda(non_blocking=True)
            reg_mask = torch.from_numpy(reg_mask.reshape(1, max_objs)).cuda(non_blocking=True)
            wh = torch.from_numpy(wh.reshape((1, max_objs, 8))).cuda(non_blocking=True)
            v = torch.from_numpy(v.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            ids = torch.from_numpy(ids.reshape((1, max_objs, 1))).cuda(non_blocking=True)
            # print(regression)

            reg_loss = self.crit_reg(
                regression, reg_mask,
                ind, wh)
            vis_loss = self.crit_vis(
                vis, reg_mask,
                ind, v)
            # print(reg)

            id_head_64 = _tranpose_and_gather_feat(identi_64, ind)
            id_head_128 = _tranpose_and_gather_feat(identi_128, ind)
            id_head_256 = _tranpose_and_gather_feat(identi_256, ind)
            id_head_end = _tranpose_and_gather_feat(identi_end, ind)
            id_head_64 = id_head_64[reg_mask > 0].contiguous()
            id_head_128 = id_head_128[reg_mask > 0].contiguous()
            id_head_256 = id_head_256[reg_mask > 0].contiguous()
            id_head_end = id_head_end[reg_mask > 0].contiguous()
            id_head_64 = self.emb_scale * torch.nn.functional.normalize(id_head_64)
            id_head_128 = self.emb_scale * torch.nn.functional.normalize(id_head_128)
            id_head_256 = self.emb_scale * torch.nn.functional.normalize(id_head_256)
            id_head_end = self.emb_scale * torch.nn.functional.normalize(id_head_end)
            id_target = ids[reg_mask > 0]
            id_weight = v[reg_mask > 0]
            id_output_64 = self.classifier_64(id_head_64).contiguous()
            id_output_128 = self.classifier_128(id_head_128).contiguous()
            id_output_256 = self.classifier_256(id_head_256).contiguous()
            id_output_end = self.classifier_end(id_head_end).contiguous()

            one_hot = torch.nn.functional.one_hot(id_target, 537).reshape(-1, 537).float()
            softmax_64 = torch.exp(id_output_64) / torch.sum(torch.exp(id_output_64), dim=1).reshape(-1, 1)
            softmax_128 = torch.exp(id_output_128) / torch.sum(torch.exp(id_output_128), dim=1).reshape(-1, 1)
            softmax_256 = torch.exp(id_output_256) / torch.sum(torch.exp(id_output_256), dim=1).reshape(-1, 1)
            softmax_end = torch.exp(id_output_end) / torch.sum(torch.exp(id_output_end), dim=1).reshape(-1, 1)

            logsoftmax_64 = torch.log(softmax_64)
            logsoftmax_128 = torch.log(softmax_128)
            logsoftmax_256 = torch.log(softmax_256)
            logsoftmax_end = torch.log(softmax_end)



            nllloss_64 = -torch.sum(id_weight * one_hot * logsoftmax_64) / id_target.shape[0]
            nllloss_128 = -torch.sum(id_weight * one_hot * logsoftmax_128) / id_target.shape[0]
            nllloss_256 = -torch.sum(id_weight * one_hot * logsoftmax_256) / id_target.shape[0]
            nllloss_end = -torch.sum(id_weight * one_hot * logsoftmax_end) / id_target.shape[0]

            # id_losses.append(self.IDLoss(id_output, id_target.squeeze(1)))
            id_losses.append(nllloss_64 + nllloss_128+nllloss_256+nllloss_end)

            vis_losses.append(vis_loss.half())
            classification_losses.append(self.crit(classification, hm).half())
            regression_losses.append(reg_loss.half())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0,
            keepdim=True), torch.stack(
            vis_losses).mean(dim=0, keepdim=True), torch.stack(
            id_losses).mean(dim=0, keepdim=True), s_det, s_id