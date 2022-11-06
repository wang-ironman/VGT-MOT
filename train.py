import time
import os

random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import copy
import argparse
import pdb
import collections
import sys
import random
from test import load_model

random.seed(random_seed)
import numpy as np

np.random.seed(random_seed)
import math
from models.model import create_model
import torch
import numpy as np
import os, shutil

np.set_printoptions(suppress=True)
from apex.fp16_utils import *
from apex import amp, optimizers

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import torch.distributed as dist
import model
from anchors import Anchors
from test import run_from_train
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, \
    PhotometricDistort, RandomSampleCrop
from torch.utils.data import Dataset, DataLoader
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

# assert torch.__version__.split('.')[1] == '4'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    # dist.init_process_group(backend='nccl')
    parser = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

    parser.add_argument('--dataset', default='csv', type=str, help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--gpus', default='0', type=str, help='Path to save the model.')
    parser.add_argument('--local_rank', default='0', type=int, help='local_rank.')
    parser.add_argument('--lr', default='5e-5', type=float, help='learning rate.')
    parser.add_argument('--model_dir', default='./exp/no_atten_half_vis_cor_6_12', type=str,
                        help='Path to save the model.')
    parser.add_argument('--root_path', default='/home/neuiva2/liweixi/data/MOT17', type=str,
                        help='Path of the directory containing both label and images')
    parser.add_argument('--csv_train', default='train_annots_transform_00_vis_half.csv', type=str,
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='train_labels.csv', type=str,
                        help='Path to file containing class list (see readme)')

    parser.add_argument('--network', help='network', type=str, default='dla34')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=8)
    parser.add_argument('--batchsize', help='Number of batch', type=int, default=10)
    parser.add_argument('--num_worker', help='Number of worker to load data', type=int, default=8)
    parser.add_argument('--resume', help='resume or not', action='store_true')
    parser.add_argument('--resume_epoch', help='resume from which epoch', type=int, default=0)
    parser.add_argument('--load_model', default='models/ctdet_coco_dla_2x.pth',
                        help='path to pretrained models')
    parser = parser.parse_args(args)
    print(parser)

    print(parser.model_dir)
    if not os.path.exists(parser.model_dir):
        os.makedirs(parser.model_dir)
    if parser.local_rank == 0:
        shutil.copyfile('models/networks/pose_dla_dcn.py', os.path.join(parser.model_dir, "model.py"))
        shutil.copyfile('train.py', os.path.join(parser.model_dir, "train.py"))

    torch.cuda.set_device(parser.local_rank)
    batchsize = int(parser.batchsize / 1)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda: {parser.local_rank}')
    # Create the data loaders
    if parser.dataset == 'csv':
        if (parser.csv_train is None) or (parser.csv_train == ''):
            raise ValueError('Must provide --csv_train when training on COCO,')

        if (parser.csv_classes is None) or (parser.csv_classes == ''):
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(parser.root_path, train_file=os.path.join(parser.root_path, parser.csv_train),
                                   class_list=os.path.join(parser.root_path, parser.csv_classes), \
                                   transform=transforms.Compose([
                                                                    transforms.ToTensor()]))  # transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        # transform=transforms.Compose([RandomSampleCrop(), PhotometricDistort(), Augmenter(),
        # Normalizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batchsize, drop_last=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    # dataloader_train = DataLoader(dataset_train,num_workers=parser.num_worker, collate_fn=collater,
    #                               batch_sampler=train_sampler, pin_memory=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, sampler=train_sampler,
                                                   collate_fn=collater, num_workers=parser.num_worker, pin_memory=True,
                                                 drop_last=True)

    # Create the model
    if parser.resume == False:
        retinanet = create_model('dla_34', {'hm': 1,'vis':1,
                                            'wh': 8}, 256)
        # center off and wh need compute 2 bounding boxes
        optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
        retinanet= load_model(retinanet, parser.load_model)
        use_gpu = True
        if use_gpu:
            retinanet = retinanet.cuda()
        # retinanet = convert_syncbn_model(retinanet).to(device)
        retinanet, optimizer = amp.initialize(retinanet, optimizer, opt_level="O1")
        # retinanet = torch.nn.DataParallel(retinanet, device_ids=gpus).to(device)

        retinanet = torch.nn.parallel.DistributedDataParallel(retinanet, device_ids=[parser.local_rank],
                                                              find_unused_parameters=True)
        retinanet.training = True
    else:
        retinanet = create_model('dla_34', {'hm': 1,
                                            'wh': 8, 'vis': 1, 'id':256}, 256)
        optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

        use_gpu = True
        if use_gpu:
            retinanet = retinanet.cuda()
        retinanet, optimizer = amp.initialize(retinanet, optimizer, opt_level="O1")
        retinanet = torch.nn.parallel.DistributedDataParallel(retinanet, device_ids=[parser.local_rank],
                                                              find_unused_parameters=True)
        retinanet, optimizer, start_epoch, amp_state = load_model(retinanet, os.path.join(parser.model_dir, 'model_{}.pt'.format(
            parser.resume_epoch)), optimizer, True, parser.lr, lr_step=[30,42])
        amp.load_state_dict(amp_state)

    
        retinanet.training = True

    # optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    cls_hist = collections.deque(maxlen=500)
    wh_hist = collections.deque(maxlen=500)
    reg_hist = collections.deque(maxlen=500)
    vis_hist = collections.deque(maxlen=500)
    id_hist = collections.deque(maxlen=500)
    retinanet.train()
    # retinanet.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))

    # (classification_loss, regression_loss, offset_loss, reid_loss) = retinanet([data['img'].cuda().float(), data['annot'], data['img_next'].cuda().float(), data['annot_next']])
    num_image = len(dataset_train)
    total_iter = 0

    for epoch_num in range(1, parser.epochs + 1):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.4f')
        losses = AverageMeter('Loss', ':.4f')
        cls_losses = AverageMeter('cls', ':.4f')
        reg_losses = AverageMeter('reg', ':.4f')
        # off_losses = AverageMeter('offset', ':.4f')
        vis_losses = AverageMeter('vis', ':.4f')
        id_losses = AverageMeter('id', ':.4f')
        train_sampler.set_epoch(epoch_num)
        progress = ProgressMeter(len(dataloader_train), [batch_time, data_time, losses, cls_losses, reg_losses, vis_losses, id_losses],
                                 prefix="Epoch: [{}]".format(epoch_num))
        retinanet.train()
        # retinanet.module.freeze_bn()

        epoch_loss = []
        end = time.time()
        for iter_num, data in enumerate(dataloader_train):
            data_time.update(time.time() - end)

            total_iter = total_iter + 1
            optimizer.zero_grad()
            (classification_loss, regression_loss, vis_loss, reid_loss, s_det, s_id) = retinanet(
                [data['img'].contiguous().cuda(non_blocking=True).half(), data['annot'],
                 data['img_next'].contiguous().cuda(non_blocking=True).half(), data['annot_next']])
            # print(regression_loss)
            classification_loss = classification_loss
            regression_loss = regression_loss * 0.1
            # offset_loss = offset_loss
            
            reid_loss = reid_loss
            # loss = classification_loss + regression_loss + track_classification_losses
            vis_loss = vis_loss * 20
            eps = 1e-5
            # classification_loss = classification_loss / (classification_loss+eps).detach()
            # regression_loss = regression_loss / (regression_loss+eps).detach()
            # reid_loss = reid_loss / (reid_loss+eps).detach()
            # vis_loss = vis_loss / (vis_loss+eps).detach()
            classification_loss = classification_loss
            regression_loss = regression_loss
            reid_loss = reid_loss
            vis_loss = vis_loss
            det_loss = classification_loss + regression_loss
            app_loss = reid_loss + vis_loss
            loss = det_loss  + app_loss
            # loss = torch.exp(-s_det) * det_loss + torch.exp(-s_id) * app_loss + (s_det + s_id)
            # loss *= 0.5

            if bool(loss == 0):
                continue

            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()


            reduced_loss = reduce_tensor(loss.data)
            reduced_classification_loss = reduce_tensor(classification_loss.data)
            reduced_regression_loss = reduce_tensor(regression_loss.data)
            reduced_vis_loss = reduce_tensor(vis_loss.data)
            # reduced_offset_loss = reduce_tensor(offset_loss.data)
            reduced_reid_loss = reduce_tensor(reid_loss.data)

            loss_hist.append(float(reduced_loss))
            cls_hist.append(float(reduced_classification_loss))
            wh_hist.append(float(reduced_regression_loss))
            vis_hist.append(float(reduced_vis_loss))
            # reg_hist.append(float(reduced_offset_loss))
            id_hist.append(float(reduced_reid_loss))
            epoch_loss.append(float(reduced_loss))

            losses.update(reduced_loss.item(), batchsize)
            cls_losses.update(reduced_classification_loss.item(), batchsize)
            reg_losses.update(reduced_regression_loss.item(), batchsize)
            # off_losses.update(reduced_offset_loss.item(), batchsize)
            vis_losses.update(reduced_vis_loss.item(), batchsize)
            id_losses.update(reduced_reid_loss.item(), batchsize)
            
            # loss_hist.append(float(loss))
            # cls_hist.append(float(classification_loss))
            # wh_hist.append(float(regression_loss))
            # off_hist.append(float(offset_loss))
            # id_hist.append(float(reid_loss))
            # epoch_loss.append(float(loss))
            
            # losses.update(loss.item(), batchsize)
            # cls_losses.update(classification_loss.item(), batchsize)
            # reg_losses.update(regression_loss.item(), batchsize)
            # off_losses.update(offset_loss.item(), batchsize)
            # id_losses.update(reid_loss.item(), batchsize)
            
            if parser.local_rank == 0:
                progress.display(iter_num)

        if parser.local_rank == 0:
            if epoch_num == 1:
                wf = open(parser.model_dir + "/log.txt", 'w')
            else:
                wf = open(parser.model_dir + "/log.txt", 'a')
            wf.write(
                'Epoch: {} | Iter: {}/{} | Cls loss: {:1.5f} | wh loss: {:1.5f} | vis loss: {:1.5f} |reid loss: {:1.5f} | total loss: {:1.5f}\n'.format(
                    epoch_num, iter_num, math.ceil(num_image / parser.batchsize), np.mean(cls_hist),
                    np.mean(wh_hist), np.mean(vis_hist), np.mean(id_hist), np.mean(loss_hist)))
            wf.close()
        if epoch_num in [30,42]:  # 25,35
            save_model(os.path.join(parser.model_dir, 'model_{}.pt'.format(epoch_num)), epoch_num, retinanet,
                       optimizer=optimizer, amp=amp)
            lr = parser.lr * (0.1 ** ([30,42].index(epoch_num) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch_num % 10 == 0 and parser.local_rank == 0:
            save_model(os.path.join(parser.model_dir, 'model_{}.pt'.format(epoch_num)), epoch_num, retinanet,
                       optimizer=optimizer, amp=amp)
        # scheduler.step(np.mean(epoch_loss))
    save_model(os.path.join(parser.model_dir, 'model_final.pt'), parser.epochs, retinanet, optimizer=optimizer, amp=amp)
    retinanet.eval()
    # torch.save(retinanet, os.path.join(parser.model_dir, 'model_final.pt'))
    run_from_train(parser.model_dir, parser.root_path)


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= torch.distributed.get_world_size()  # 总进程数
    # print(torch.distributed.get_world_size())
    return rt


def save_model(path, epoch, model, optimizer=None, amp=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    if not (amp is None):
        data['amp'] = amp.state_dict()
    torch.save(data, path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
