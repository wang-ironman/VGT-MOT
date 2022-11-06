# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import numpy as np
import torchvision
import time
import math
import os
import copy
import pdb
import argparse
import sys
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage
import torch
import motmetrics as mm
import model
from collections import OrderedDict

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
from models.decode import mot_decode, mot_decode_4, mot_decode_4_new, _nms
from dataloader import letterbox_test
from cython_bbox import bbox_overlaps as bbox_ious
import lap
import numpy as np
from collections import deque
import scipy
from scipy.spatial.distance import cdist
from tools.tracking_utils.kalman_filter import KalmanFilter
from tools.tracking_utils import kalman_filter as KF
from models.model import create_model
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, \
    RGB_MEAN, RGB_STD
from scipy.optimize import linear_sum_assignment

from tools.tracking_utils.utils import *
from tools.tracking_utils import visualization as vis
from tools.tracking_utils.timer import Timer
from tools.tracking_utils.evaluation import Evaluator_half
from models.utils import _tranpose_and_gather_feat
from tools.tracking_utils.utils import mkdir_if_missing

# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255),
              (0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255),
              (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, tlwh_next, score, temp_feat, vis, pos_det_next, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self._tlwh_next = np.asarray(tlwh_next, dtype=np.float)
        self._vis = vis
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.pos_next = pos_det_next
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.New
        self.smooth_feat = None
        self.update_features(temp_feat, vis)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    # def update_features(self, feat, det_vis):
    #     feat /= np.linalg.norm(feat)
    #     self.curr_feat = feat
    #     if self.smooth_feat is None:
    #         self.smooth_feat = feat
    #     else:
    #         self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    #     self.features.append(feat)
    #     self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_features(self, feat, det_vis):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # if det_vis > self._vis:
            #     self._vis = det_vis
            #     self.smooth_feat = feat
            self.smooth_feat = (1-det_vis)*self.smooth_feat + (det_vis)*feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, is_Start=False):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if is_Start:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(new_track._tlwh))
        self._tlwh_next = new_track.tlwh_next
        self.pos_next = new_track.pos_next
        # self.update_features(new_track.curr_feat)
        self.update_features(new_track.curr_feat, new_track._vis)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh_next = new_track.tlwh_next
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.pos_next = new_track.pos_next
        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat, new_track._vis)
            # self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlwh_next(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            # self.mean = self._tlwh_next.copy()
            return self._tlwh_next.copy()
        if self.state == TrackState.Tracked:
            return self._tlwh_next.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        # print(ret)
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def tlbr_next(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh_next.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance_for_remove(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919}
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def run_each_dataset(model_dir, retinanet, dataset_path, subset, cur_dataset, frame_rate, det_path):
    print(cur_dataset)

    img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset, 'img1'))
    img_list = [os.path.join(dataset_path, subset, cur_dataset, 'img1', _) for _ in img_list if
                ('jpg' in _) or ('png' in _)]
    img_list = sorted(img_list)
    img_len = len(img_list)
    last_feat = None

    confidence_threshold = 0.4
    IOU_threshold = 0.5
    retention_threshold = frame_rate

    tracked_stracks_all = []  # type: list[STrack]
    lost_stracks_all = []  # type: list[STrack]
    removed_stracks_all = []  # type: list[STrack]
    kalman_filter = KalmanFilter()
    results = []
    max_id = 0
    max_draw_len = 100
    draw_interval = 5
    if cur_dataset == 'MOT17-05':
        img_width = 640
        img_height = 480
    else:
        img_width = 1920
        img_height = 1080
    fps = 20
    timer = Timer()
    id_dict = {}
    out_video = os.path.join(model_dir, 'result_half', cur_dataset + '.mp4')
    videoWriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (img_width, img_height))
    out_video1 = os.path.join(model_dir, 'result_half_line', cur_dataset + '.mp4')
    videoWriter1 = cv2.VideoWriter(out_video1, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                   (img_width, img_height))
    for idx in range(img_len // 2, img_len + 1):
        i = idx - 1
        # print('tracking: ', i)
        if idx % 20 == 0:
            print('Processing frame {}/{} ({:.2f} fps)'.format(cur_dataset, idx, 1. / max(1e-5, timer.average_time)))
        with torch.no_grad():
            timer.tic()
            data_path1 = img_list[min(idx, img_len - 1)]
            if idx > img_len // 2:
                old_img = img_origin1
            img_origin1 = cv2.imread(data_path1)
            activated_starcks = []
            refind_stracks = []
            lost_stracks = []
            removed_stracks = []
            # print(img_origin1.shape)
            img_h, img_w, img_c = img_origin1.shape
            img_height, img_width = img_h, img_w
            img_blob, ratio, dw, dh = letterbox_test(img_origin1)
            mean = np.array([0.40789654, 0.44719302, 0.47026115],  # BGR
                            dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.28863828, 0.27408164, 0.27809835],
                           dtype=np.float32).reshape(1, 1, 3)
            # img_blob = img_blob[:, :, ::-1]  # to RGB.
            img_blob = np.ascontiguousarray(img_blob, dtype=np.float32)
            img_blob /= 255.0
            img_blob = (img_blob - mean) / std

            # cv2.imwrite("./data/{0}.jpg".format(idx), img_blob)
            # print(img1.shape)
            # for name in retinanet.state_dict():
            #     print(name)
            width = img_w
            height = img_h
            inp_height = 608
            inp_width = 1088
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
            meta = {'c': c, 's': s,
                    'out_height': inp_height // 4,
                    'out_width': inp_width // 4}

            img_blob = torch.from_numpy(img_blob)
            img_blob = img_blob.unsqueeze(0).permute(0, 3, 1, 2)
            cls_feat, reg_feat, vis_feat, id_feat, last_feat = retinanet(img_blob.cuda().float(), last_feat=last_feat)

            hm = cls_feat.sigmoid_()
            vis_feat = vis_feat.sigmoid_()
            wh = reg_feat
            # id_feature = reid_feat
            # reid = id_feature.sigmoid_()
            # id_feature = F.normalize(id_feature, dim=1)
            # reg = off_feat
            if idx > img_len // 2:
                hm = hm.reshape(1, 1, inp_height // 4, inp_width // 4)
                # print(torch.max(hm))
                wh = wh.reshape(1, 8, inp_height // 4, inp_width // 4)
                # vis = vis_feat.reshape(1, 1, inp_height // 4, inp_width // 4)
                # reg = reg.reshape(1, 4, inp_height // 4, inp_width // 4)
                # reid = id_feature.reshape(1, 1, inp_height // 4, inp_width // 4)
                # print(torch.max(reid))
                wh_t = wh[:, 0:4, :, :]
                wh_t1 = wh[:, 4:, :, :]
                # reg_t = reg[:, 0:2, :, :]
                # reg_t1 = reg[:, 2:, :, :]

                # print(wh[:, :, 0:20, 0])
                # print(reg[:, :, 0, 0])
                dets_t, inds_t = mot_decode_4_new(hm, wh_t, K=128)  # dets的坐标是相对于heatmap
                dets_t1, inds_t1 = mot_decode_4_new(hm, wh_t1, K=128)
                pos_t1 = dets_t1[0].cpu().numpy() # (128, 6)

                # dets_t, inds_t = mot_decode_4(hm, wh_t, reg=reg_t, K=128)  # dets的坐标是相对于heatmap
                # dets_t1, inds_t1 = mot_decode_4(hm, wh_t1, reg=reg_t1, K=128)
                id_feat = torch.nn.functional.normalize(id_feat, dim=1)
                id_feature = _tranpose_and_gather_feat(id_feat, inds_t)
                id_feature = id_feature.squeeze(0)
                id_feature = id_feature.cpu().numpy()

                vis_feature = _tranpose_and_gather_feat(vis_feat, inds_t)
                vis_feature = vis_feature.squeeze(0)
                vis_feature = vis_feature.cpu().numpy()

                dets_t = post_process(dets_t, meta)  # 转换成针对原图的坐标

                dets_t = merge_outputs([dets_t], K=128)[1]

                dets_t1 = post_process(dets_t1, meta)  # 转换成针对原图的坐标
                
                dets_t1 = merge_outputs([dets_t1], K=128)[1]
                
                dets = np.concatenate([dets_t, dets_t1], axis=1)
                remain_inds = dets[:, 4] > confidence_threshold
                dets = dets[remain_inds]
                pos_det_next = pos_t1[remain_inds]
                pos_det_next = pos_det_next[:, :4]

                for i in range(dets.shape[0]):
                    det_path.write(cur_dataset + "/" + str(idx).zfill(6) + " " +
                                   str(dets[i, 4]) + " " +
                                   str(dets[i, 0]) + " " +
                                   str(dets[i, 1]) + " " +
                                   str(dets[i, 2]) + " " +
                                   str(dets[i, 3]) + "\n")

                id_feature = id_feature[remain_inds]
                vis_feature = vis_feature[remain_inds]
                if len(dets) > 0:  # det都初始成track
                    '''Detections'''
                    detections = [
                        STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), STrack.tlbr_to_tlwh(tlbrs[5:9]), tlbrs[4], f, vis_f, p, 30)
                        for
                        (tlbrs, f, vis_f, p) in zip(dets[:, :], id_feature, vis_feature, pos_det_next)]
                else:
                    detections = []
                # imgs = old_img
                # # imgs_next = img_origin1
                # if not os.path.exists(os.path.join(model_dir, 'result_half', cur_dataset + "_det")):
                #     os.mkdir(os.path.join(model_dir, 'result_half', cur_dataset + "_det"))
                # for det in detections:
                #     tlwh = det.tlwh
                #     # print(tlwh)
                #     x1 = int(tlwh[0])
                #     y1 = int(tlwh[1])
                #     x2 = int(tlwh[0] + tlwh[2])
                #     y2 = int(tlwh[1] + tlwh[3])
                #     c1 = random.randint(0,255)
                #     c2 = random.randint(0, 255)
                #     c3 = random.randint(0, 255)
                #
                #     tlwh_next = det.tlwh_next
                #     # print(tlwh)
                #     x1_n = int(tlwh_next[0])
                #     y1_n = int(tlwh_next[1])
                #     x2_n = int(tlwh_next[0] + tlwh_next[2])
                #     y2_n = int(tlwh_next[1] + tlwh_next[3])
                #
                #     cv2.rectangle(imgs, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                #     # cv2.rectangle(imgs_next, (x1_n, y1_n), (x2_n, y2_n), color=(0, 255, 0), thickness=2)
                #     cv2.putText(imgs, str(det._vis), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                #
                # cv2.imwrite(os.path.join(model_dir, 'result_half', cur_dataset + "_det") + "/" + str(idx) + ".jpg", imgs)
                # continue
                ''' Add newly detected tracklets to tracked_stracks'''
                unconfirmed = []
                tracked_stracks = []  # type: list[STrack]
                if (idx - 1) == img_len // 2:
                    # print("1")
                    for inew in range(len(detections)):
                        track = detections[inew]
                        if track.score < confidence_threshold:
                            continue
                        track.activate(kalman_filter, idx, is_Start=True)
                        tracked_stracks_all.append(track)
                    online_targets = tracked_stracks_all

                    online_tlwhs = []
                    online_ids = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > 100:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                    timer.toc()
                    # save results
                    results.append((idx, online_tlwhs, online_ids))

                    online_im = vis.plot_tracking(old_img, online_tlwhs, online_ids, frame_id=idx,
                                                  fps=1. / timer.average_time)
                    if not os.path.exists(os.path.join(model_dir, 'result_half', cur_dataset)):
                        os.mkdir(os.path.join(model_dir, 'result_half', cur_dataset))
                    # print(os.path.join(model_dir, 'result_half', cur_dataset, '{:05d}.jpg'.format(idx)))
                    # print(online_im.shape)
                    cv2.imwrite(os.path.join(model_dir, 'result_half', cur_dataset, '{:05d}.jpg'.format(idx)),
                                online_im)
                    videoWriter.write(online_im)
                    continue
                for track in tracked_stracks_all:
                    if track.state == TrackState.New:
                        unconfirmed.append(track)
                    if track.state == TrackState.Tracked:
                        tracked_stracks.append(track)

                ''' Step 2: First association, with embedding'''
                strack_pool = joint_stracks(tracked_stracks, lost_stracks_all)
                # strack_pool = tracked_stracks
                # strack_pool_lost = lost_stracks_all
                STrack.multi_predict(strack_pool)
                # STrack.multi_predict(strack_pool_lost)
                # dists = embedding_distance(strack_pool, detections)
                dists = mix_distance(strack_pool, detections)
                # dists = iou_distance(strack_pool, detections)
                # dists = gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
                dists = fuse_motion(kalman_filter, dists, strack_pool, detections)
                matches, u_track, u_detection = linear_assignment(dists, thresh=0.7)
                for itracked, idet in matches:
                    track = strack_pool[itracked]
                    det = detections[idet]
                    if track.state == TrackState.Tracked:
                        track.update(detections[idet], idx)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, idx, new_id=False)
                        refind_stracks.append(track)

                ''' Step 3: Second association, with IOU'''
                detections = [detections[i] for i in u_detection]
                r_tracked_stracks = [strack_pool[i] for i in u_track]  # why
                dists = iou_distance(r_tracked_stracks, detections)
                matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
                for itracked, idet in matches:
                    track = r_tracked_stracks[itracked]
                    det = detections[idet]
                    if track.state == TrackState.Tracked:
                        print("2")
                        track.update(det, idx)
                        activated_starcks.append(track)
                    # else:
                    #     print("1")
                    #     track.re_activate(det, idx, new_id=False)
                    #     refind_stracks.append(track)

                for it in u_track:
                    track = r_tracked_stracks[it]
                    if not track.state == TrackState.Lost:
                        pos_next = track.pos_next
                        pos_x = (pos_next[0] + pos_next[2]) / 2
                        pos_y = (pos_next[1] + pos_next[3]) / 2
                        if 0 < pos_x < 272 and 0 < pos_y < 152:
                            hm_new = _nms(hm)
                            # print(hm_new)
                            pos_x = int(pos_x)
                            pos_y = int(pos_y)
                            f = False
                            for i in range(pos_x - 1, pos_x + 1):
                                for j in range(pos_y - 1, pos_y + 1):
                                    if 0 < i < 272 and 0 < j < 152:
                                        if hm_new[0, 0, j, i] != 0:
                                            x = i
                                            y = j
                                            f = True
                            if f == False :
                                track.mark_lost()
                                lost_stracks.append(track)
                                continue
                            score_next = hm_new[0, 0, y, x]
                            idf = id_feat[0, :, y, x].cpu().numpy()
                            visf = vis_feat[0, 0, y, x].cpu().numpy()

                            wh1 = wh[0, :, y, x]
                            wh_t = wh1[:4].cpu().numpy()
                            wh_t1 = wh1[4:].cpu().numpy()
                            pos_x = x
                            pos_y = y
                            det_t = [pos_x - wh_t[0], pos_y - wh_t[1], pos_x + wh_t[2], pos_y + wh_t[3]]
                            det_t1 = [pos_x - wh_t1[0], pos_y - wh_t1[1], pos_x + wh_t1[2], pos_y + wh_t1[3]]
                            old_det_t1 = det_t1
                            
                            det_t = [pos_x - wh_t[0], pos_y - wh_t[1], pos_x + wh_t[2], pos_y + wh_t[3], score_next, 0.0]
                            det_t1 = [pos_x - wh_t1[0], pos_y - wh_t1[1], pos_x + wh_t1[2], pos_y + wh_t1[3], score_next, 0.0]
                            det_t = torch.from_numpy(np.array(det_t, dtype=np.float32)).cpu().cuda()
                            det_t1 = torch.from_numpy(np.array(det_t1, dtype=np.float32)).cpu().cuda()
                            det_t = det_t.view(1, 1, 6)
                            det_t1 = det_t1.view(1, 1, 6)
                            det_t = post_process(det_t, meta)
                            det_t1 = post_process(det_t1, meta)

                            det_t = det_t.get(1)
                            det_t1 = det_t1.get(1)

                            if score_next > 0.2 and track.tracklet_len >= 3:
                                new_track = STrack(STrack.tlbr_to_tlwh(det_t[0][0:4]), STrack.tlbr_to_tlwh(det_t1[0][0:4]), score_next, idf, visf, old_det_t1, 30)
                                track.update(new_track, idx)
                                activated_starcks.append(new_track)
                                continue
                        track.mark_lost()
                        lost_stracks.append(track)

                '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
                detections = [detections[i] for i in u_detection]
                dists = iou_distance(unconfirmed, detections)
                matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.5)
                for itracked, idet in matches:
                    unconfirmed[itracked].update(detections[idet], idx)
                    activated_starcks.append(unconfirmed[itracked])
                for it in u_unconfirmed:
                    track = unconfirmed[it]
                    track.mark_removed()
                    removed_stracks.append(track)

                """ Step 4: Init new stracks"""
                for inew in u_detection:
                    track = detections[inew]
                    if track.score < confidence_threshold:
                        continue
                    track.activate(kalman_filter, idx, is_Start=False)
                    activated_starcks.append(track)
                """ Step 5: Update state"""
                for track in lost_stracks_all:
                    if idx - track.end_frame > retention_threshold:
                        track.mark_removed()
                        removed_stracks.append(track)

                # print('Ramained match {} s'.format(t4-t3))

                tracked_stracks_all = [t for t in tracked_stracks_all if t.state == TrackState.Tracked]
                tracked_stracks_all = joint_stracks(tracked_stracks_all, activated_starcks)
                tracked_stracks_all = joint_stracks(tracked_stracks_all, refind_stracks)
                lost_stracks_all = sub_stracks(lost_stracks_all, tracked_stracks_all)
                lost_stracks_all.extend(lost_stracks)
                lost_stracks_all = sub_stracks(lost_stracks_all, removed_stracks_all)
                removed_stracks.extend(removed_stracks)
                tracked_stracks_all, lost_stracks_all = remove_duplicate_stracks(tracked_stracks_all, lost_stracks_all)
                # get scores of lost tracks
                online_targets = [track for track in tracked_stracks_all if track.is_activated]

                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 100:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                timer.toc()
                # save results
                results.append((idx, online_tlwhs, online_ids))

                online_im = vis.plot_tracking(old_img, online_tlwhs, online_ids, frame_id=idx,
                                              fps=1. / timer.average_time)
                if not os.path.exists(os.path.join(model_dir, 'result_half', cur_dataset)):
                    os.mkdir(os.path.join(model_dir, 'result_half', cur_dataset))
                # print(os.path.join(model_dir, 'result_half', cur_dataset, '{:05d}.jpg'.format(idx)))
                # print(online_im.shape)
                cv2.imwrite(os.path.join(model_dir, 'result_half', cur_dataset, '{:05d}.jpg'.format(idx)), online_im)
                videoWriter.write(online_im)

                # print('saving: ', i)
                img = old_img

                for j in range(len(online_targets)):
                    x, y, w, h = online_targets[j].tlwh
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    trace_id = online_targets[j].track_id

                    id_dict.setdefault(str(trace_id), []).append((int((x1 + x2) / 2), y2))
                    draw_trace_id = str(trace_id)
                    draw_caption(img, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)

                    trace_len = len(id_dict[str(trace_id)])
                    trace_len_draw = min(max_draw_len, trace_len)

                    for k in range(trace_len_draw - draw_interval):
                        if (k % draw_interval == 0):
                            draw_point1 = id_dict[str(trace_id)][trace_len - k - 1]
                            draw_point2 = id_dict[str(trace_id)][trace_len - k - 1 - draw_interval]
                            cv2.line(img, draw_point1, draw_point2, color=color_list[trace_id % len(color_list)],
                                     thickness=2)

                # cv2.imwrite(os.path.join(save_img_dir, str(i + 1).zfill(6) + '.jpg'), img)
                videoWriter1.write(img)
                cv2.waitKey(0)
        # save results
    write_results(os.path.join(model_dir, 'result_half', cur_dataset + '.txt'), results, 'mot')
    videoWriter1.release()
    videoWriter.release()
    return img_len, timer.average_time, timer.calls, os.path.join(model_dir, 'result_half', cur_dataset + '.txt')


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    print('save results to {}'.format(filename))


def draw_caption(image, box, caption, color):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def run_from_train(model_dir, root_path):
    if not os.path.exists(os.path.join(model_dir, 'result_half')):
        os.makedirs(os.path.join(model_dir, 'result_half'))
    if not os.path.exists(os.path.join(model_dir, 'result_half_line')):
        os.makedirs(os.path.join(model_dir, 'result_half_line'))

    retinanet = torch.load(os.path.join(model_dir, 'model_final.pt'))

    use_gpu = True

    if use_gpu: retinanet = retinanet.cuda()

    retinanet.eval()

    for seq_num in [2, 4, 5, 9, 10, 11, 13]:
        run_each_dataset(model_dir, retinanet, root_path, 'train', 'MOT17-{:02d}'.format(seq_num))
    for seq_num in [1, 3, 6, 7, 8, 12, 14]:
        run_each_dataset(model_dir, retinanet, root_path, 'test', 'MOT17-{:02d}'.format(seq_num))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
    parser.add_argument('--dataset_path', default='/home/neuiva2/liweixi/data/MOT17/', type=str,
                        help='Dataset path, location of the images sequence.')
    parser.add_argument('--model_dir', default='./ctracker_free_all/', help='Path to model (.pt) file.')
    parser.add_argument('--epoch', default='final', help='choose one epoch to test')
    parser = parser.parse_args(args)

    if not os.path.exists(os.path.join(parser.model_dir, 'result_half')):
        os.makedirs(os.path.join(parser.model_dir, 'result_half'))
    if not os.path.exists(os.path.join(parser.model_dir, 'result_half_line')):
        os.makedirs(os.path.join(parser.model_dir, 'result_half_line'))
    # retinanet = torch.load(os.path.join(parser.model_dir, 'model_final.pt'))
    retinanet = create_model('dla_34', {'hm': 1, 'vis': 1,
                                        'wh': 8}, 256)
    retinanet = load_model(retinanet, os.path.join(parser.model_dir, 'model_{}.pt'.format(parser.epoch)))
    use_gpu = True
    # for name, param in retinanet.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print(retinanet.state_dict()['vis.4.weight'])
    # print(retinanet.state_dict()['vis.4.bias'])
    if use_gpu: retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.eval()
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    frame_rates = {2: 30, 4: 30, 5: 14, 9: 30, 10: 30, 11: 30, 13: 25}
    det_file_path = os.path.join(parser.model_dir, "MOT_det.txt")
    f = open(det_file_path, 'a+')
    for seq_num in [2, 4, 5, 9, 10, 11, 13]:
        nf, ta, tc, result_filename = run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'train',
                                                       'MOT17-{:02d}'.format(seq_num),
                                                       frame_rate=frame_rates.get(seq_num), det_path=f)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        seq_name = "MOT17-" + str(seq_num).zfill(2)
        print('Evaluate seq: {}'.format(seq_name))

        output_dir = os.path.join(parser.model_dir, 'result_half', seq_name)
        evaluator = Evaluator_half("/home/neuiva2/liweixi/data/MOT17/train", seq_name, 'mot')
        accs.append(evaluator.eval_file(result_filename))
        # output_dir = os.path.abspath(output_dir)
        # output_video_path = os.path.join(output_dir, '{}.mp4'.format(seq_name))
        # cmd_str = 'ffmpeg -f image2 -i {}/%5d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        # print(cmd_str)
        # os.system(cmd_str)
    f.close()
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    print('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    # x = ["MOT17-04"]
    x = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"
        , "MOT17-11", "MOT17-13"]
    summary = Evaluator_half.get_summary(accs, x, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator_half.save_summary(summary, os.path.join(parser.model_dir, 'summary_{}.xlsx'.format("MOT17_train")))
    # for seq_num in [1, 3, 6, 7, 8, 12, 14]:
    #     run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'test', 'MOT17-{:02d}'.format(seq_num))


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # print("="*80)
    # print(state_dict_)
    # print("=" * 80)
    # convert data_parallal to models
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:

            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created models parameters
    msg = 'If you see this, your models does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    # print(model_state_dict)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 1)
    for j in range(1, 1 + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(detections, K=128):
    result_half = {}
    for j in range(1, 1 + 1):
        result_half[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [result_half[j][:, 4] for j in range(1, 1 + 1)])
    if len(scores) > K:
        kth = len(scores) - K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 1 + 1):
            keep_inds = (result_half[j][:, 4] >= thresh)
            result_half[j] = result_half[j][keep_inds]
    return result_half


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance_for_remove(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print("111")
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # print(atlbrs)
    # print(btlbrs)
    # print("-"*80)
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
        print("111")
    else:
        atlbrs = [track.tlbr_next for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    # print(atlbrs)
    # print(btlbrs)
    # print("-"*80)
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def mix_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix_reid = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features

    atlbrs = [track.tlbr_next for track in tracks]
    btlbrs = [track.tlbr for track in detections]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix_iou = 1 - _ious
    vis = [track._vis for track in detections]
    vis = np.stack(vis, axis=1)
    cost_matrix = vis * cost_matrix_reid + (1 - vis) * cost_matrix_iou
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


if __name__ == '__main__':
    main()
