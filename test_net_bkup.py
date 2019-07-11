# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import _init_paths
import cv2
from torch.autograd import Variable
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.vgg16_4ch import vgg16_4ch
from model.faster_rcnn.vgg16_5ch import vgg16_5ch
from model.faster_rcnn.resnet import resnet
from sorting_models import sort_models

import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
from datasets.imdb import imdb
from datasets.imdb import ROOT_DIR
from datasets import ds_utils
from datasets.kinect_fruits_eval import voc_eval
#from .voc_eval import voc_eval

from model.utils.config import cfg
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16_5ch.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, vgg16_5ch, res50, res101, res152',
                      default='vgg16_5ch', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="/work/jgene/faster_rcnn/data/kinect_fruits_models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  # Color space
  parser.add_argument('--RGB', dest='RGB',
                      help='Evaluation RGB images',
                      action='store_true')
  parser.add_argument('--NIR', dest='NIR',
                      help='Evaluation NIR images',
                      action='store_true')
  parser.add_argument('--DEPTH', dest='DEPTH',
                      help='Evaluation DEPTH images',
                      action='store_true')

  parser.add_argument('--ovthresh', action='append')
  parser.add_argument('--minconfid', action='append')
  parser.add_argument('--anchor', action='append')
  parser.add_argument('--anchor_ratio', action='append')


  args = parser.parse_args()
  return args


def get_kinect_fruits_results_file_template(self,cls2save, session):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 's'+ str(session) + '_det_' + cls2save + '.txt'
    filedir = os.path.join(self._devkit_path, 'apples', 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    path = os.path.join(filedir, filename)

    return path

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "kinect_fruits":
      args.imdb_name = "train"
      args.imdbval_name = "val"
      args.imdbtest_name = "test"
      #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '100']
      #args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '100']
      args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', list(np.array(args.anchor_ratio).astype(np.float)), 'MAX_NUM_GT_BOXES', '100']
  elif args.dataset == "kinect_fruits_k":
      args.imdb_name = "train_k"
      args.imdbval_name = "val_k"
      args.imdbtest_name = "test_k"
      #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '100']
      #args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '100']
      args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', list(np.array(args.anchor_ratio).astype(np.float)), 'MAX_NUM_GT_BOXES', '100']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbtest_name, training = False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg16_4ch':
    fasterRCNN = vgg16_4ch(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg16_5ch':
    fasterRCNN = vgg16_5ch(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)
  
  fasterRCNN.create_architecture()
  start = time.time()
  files = sort_models(input_dir,args.checksession,args.checkpoint)
  epoch = 0
  meanAP_epoch = np.zeros((len(files),len(args.ovthresh),len(args.minconfid)))
  recall = np.zeros((len(files),len(args.ovthresh),len(args.minconfid)))
  precision = np.zeros((len(files),len(args.ovthresh),len(args.minconfid)))
  meanTime = np.zeros((len(files),len(args.ovthresh),len(args.minconfid)))
  print('Mean ap vector:\n')
  print(meanAP_epoch)
  
  for file in files:
    load_dir = os.path.join(input_dir,file)
    print("load checkpoint %s" % (load_dir))
    checkpoint = torch.load(load_dir)
    # if len(list(np.array(args.anchor_ratio).astype(np.float)))==1:
    #     indices_checkpoint = torch.cuda.LongTensor([3, 4, 5, 12, 13, 14])
    #     checkpoint['model']['RCNN_rpn.RPN_cls_score.weight'] = torch.index_select(checkpoint['model']['RCNN_rpn.RPN_cls_score.weight'], 0, indices_checkpoint)
    #     checkpoint['model']['RCNN_rpn.RPN_cls_score.bias'] = torch.index_select(checkpoint['model']['RCNN_rpn.RPN_cls_score.bias'], 0, indices_checkpoint)
    #     #checkpoint['model']['RCNN_rpn.RPN_bbox_pred.weight'] = torch.index_select(checkpoint['model']['RCNN_rpn.RPN_bbox_pred.weight'], 0, indices_checkpoint)
    #     #checkpoint['model']['RCNN_rpn.RPN_bbox_pred.bias'] = torch.index_select(checkpoint['model']['RCNN_rpn.RPN_bbox_pred.bias'], 0, indices_checkpoint)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    
    print('load model successfully!')

    if args.cuda:
      cfg.CUDA = True
  
    if args.cuda:
      fasterRCNN.cuda()
  

    max_per_image = 100
  
    vis = args.vis
  
    if vis:
      thresh = 0.05
    else:
      thresh = 0.00
  
    save_name = 'faster_rcnn_'+str(args.checksession)
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
  
    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                          imdb.num_classes, args.RGB, args.NIR, args.DEPTH, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=True)
  
    data_iter = iter(dataloader)
  
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections_'+str(args.checksession)+'.pkl')
  
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    InferTime = np.zeros(num_images)
    for i in range(num_images):
  
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
  
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
  
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
  
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(args.batch_size, -1, 4)
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))
  
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
  
        pred_boxes /= data[1][0][2]
  
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_dets, cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
              all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
              all_boxes[j][i] = empty_array
  
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
  
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
  
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()
  
        if vis:
            cv2.imwrite('result.png', im2show)
            #pdb.set_trace()
            #cv2.imshow('test', im2show)
            #cv2.waitKey(0)
        InferTime[i] = misc_toc - det_tic
    print('dump_all_boxes:')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('finish dump_all_boxes')

    for cls_ind, cls in enumerate(imdb.classes):
        if cls == '__background__':
            continue
        print('Writing {} kinect_fruits results file'.format(cls))

        filename = get_kinect_fruits_results_file_template(imdb,cls,args.checksession)

        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(imdb.image_index):
                #print('fallo 1')

                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue

                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    annopath = os.path.join(
        imdb._devkit_path, 'apples', 'square_annotations1',
        '{:s}.xml')
    print(annopath)

    imagesetfile = os.path.join(
        imdb._devkit_path, 'apples', 'sets', imdb._image_set + '.txt')
    cachedir = os.path.join(imdb._devkit_path, 'annotations_cache')
    aps = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print('Evaluating detections')
    IoUidx=-1

    for IoU in np.array(args.ovthresh).astype(np.float):
        IoUidx=IoUidx+1
        confididx = -1
        for confid in np.array(args.minconfid).astype(np.float):
            #confids=np.array(args.minconfid).astype(np.float)
            #confid = confids[2]
            confididx=confididx+1

            #meanAP_epoch[epoch], recall[epoch], precision[epoch] = imdb.evaluate_detections(all_boxes, output_dir)
            ########## New Eval

            # for cls_ind, cls in enumerate(imdb.classes):
            #     if cls == '__background__':
            #         continue
            #     print('Writing {} kinect_fruits results file'.format(cls))
            #
            #     filename = get_kinect_fruits_results_file_template(imdb,cls,args.checksession)
            #
            #     with open(filename, 'wt') as f:
            #         for im_ind, index in enumerate(imdb.image_index):
            #             #print('fallo 1')
            #
            #             dets = all_boxes[cls_ind][im_ind]
            #             if dets == []:
            #                 continue
            #
            #             # the VOCdevkit expects 1-based indices
            #             for k in xrange(dets.shape[0]):
            #                 f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
            #                         format(index, dets[k, -1],
            #                                dets[k, 0] + 1, dets[k, 1] + 1,
            #                                dets[k, 2] + 1, dets[k, 3] + 1))

            #print('fallo 2')
            # annopath = os.path.join(
            #     imdb._devkit_path, 'apples', 'square_annotations1',
            #     '{:s}.xml')
            # print(annopath)
            #
            # imagesetfile = os.path.join(
            #     imdb._devkit_path, 'apples', 'sets', imdb._image_set + '.txt')
            # cachedir = os.path.join(imdb._devkit_path, 'annotations_cache')
            # aps = []
            #
            # if not os.path.isdir(output_dir):
            #     os.mkdir(output_dir)
            for i, cls in enumerate(imdb._classes):
                # print(i)
                # print(cls)
                if cls == '__background__':
                    continue
                #print('fallo 3')
                filename = get_kinect_fruits_results_file_template(imdb, cls, args.checksession)
                #print('fallo 4')

                # detpath=filename
                # classname=cls
                # ovthresh=IoU
                # minconfid=confid
                # use_07_metric = False

                #detpath='/work/jgene/faster_rcnn/data/kinect_fruits_dataset/apples/results/s67_det_Poma.txt'
                #annopath='/work/jgene/faster_rcnn/data/kinect_fruits_dataset/apples/square_annotations1/{:s}.xml'
                #imagesetfile='/work/jgene/faster_rcnn/data/kinect_fruits_dataset/apples/sets/test.txt'
                #classname = 'Poma'
                #cachedir='/work/jgene/faster_rcnn/data/kinect_fruits_dataset/annotations_cache'
                #ovthresh=0.4
                #minconfid=0.15
                #use_07_metric = False
                print(filename)
                print(annopath)
                print(imagesetfile)
                print(cls)
                print(cachedir)
                print(IoU)
                print(confid)

                rec, prec, ap, r, p = voc_eval(filename, annopath, imagesetfile, cls, cachedir,  IoU, confid)
                #rec, prec, ap, r, p = voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh,minconfid)
                #print('fallo 5')
                # rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.2)
                aps += [ap]
                print('AP for {} = {:.4f}'.format(cls, ap))
                print('min IoU = {:.2f} ; Confidence = {:.2f}'.format(IoU, confid))
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            #print('fallo 6')
            meanAP = np.mean(aps)
            meanTime[epoch,IoUidx,confididx] = np.mean(InferTime)
            print('Mean AP = {:.4f}'.format(meanAP))
            print('Recall = {:.4f}'.format(r))
            print('Precision = {:.4f}'.format(p))
            print('Mean Time = {:.4f}'.format(meanTime[epoch,IoUidx,confididx]))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
            print('-- Thanks, The Management')
            print('--------------------------------------------------------------')
            print('eval_ap:')
            print(meanAP)

            # if imdb.config['matlab_eval']:
            #     imdb._do_matlab_eval(output_dir)
            # if imdb.config['cleanup']:
            #     for cls in imdb._classes:
            #         print('cls:')
            #         print(cls)
            #         if cls == '__background__':
            #             continue
            #         filename = imdb.get_kinect_fruits_results_file_template().format(cls)
            #         os.remove(filename)
            #######
            meanAP_epoch[epoch,IoUidx,confididx]=meanAP
            recall[epoch,IoUidx,confididx]=r
            precision[epoch,IoUidx,confididx]=p
            # if not os.path.isdir('output'):
            #     os.mkdir('output')
            # if not os.path.isdir('output/test_s'+str(args.checksession)):
            #     os.mkdir('output/test_s'+str(args.checksession))
            # if epoch == 0:
            #   pickle.dump(meanAP_epoch[:,IoUidx,confididx], open('output/test_s'+str(args.checksession)+'/meanAP_s'+str(args.checksession)+'_epoch_'+str(epoch+1)+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)
            #
            #
            #
            # mat = np.matrix(meanAP_epoch[:,IoUidx,confididx])
            # with open('output/test_s'+str(args.checksession)+'/meanAP_s'+str(args.checksession)+'_IoU_'+str(IoU)[2:]+'_confid_' +str(confid)[2:]+ '.txt', 'wb') as f:
            #     for line in mat:
            #         np.savetxt(f, line, fmt='%.4f')
            # mat = np.matrix(recall[:,IoUidx,confididx])
            # with open('output/test_s'+str(args.checksession)+'/recall_s'+str(args.checksession)+'_IoU_'+str(IoU)[2:]+'_confid_' +str(confid)[2:]+ '.txt', 'wb') as f:
            #     for line in mat:
            #         np.savetxt(f, line, fmt='%.4f')
            # mat = np.matrix(precision[:,IoUidx,confididx])
            # with open('output/test_s'+str(args.checksession)+'/precision_s'+str(args.checksession)+'_IoU_'+str(IoU)[2:]+'_confid_' +str(confid)[2:]+ '.txt', 'wb') as f:
            #     for line in mat:
            #         np.savetxt(f, line, fmt='%.4f')
            # mat = np.matrix(meanTime[:,IoUidx,confididx])
            # with open('output/test_s'+str(args.checksession)+'/meanTime_s'+str(args.checksession)+'_IoU_'+str(IoU)[2:]+'_confid_' +str(confid)[2:]+ '.txt', 'wb') as f:
            #     for line in mat:
            #         np.savetxt(f, line, fmt='%.4f')

    print("End of epoch {}".format(epoch + 1))
    epoch = epoch + 1
    print("Start of epoch {}".format(epoch + 1))
    end = time.time()
    print("test time: %0.4fs" % (end - start))


IoUidx=-1

for IoU in np.array(args.ovthresh).astype(np.float):
    IoUidx=IoUidx+1
    if not os.path.isdir('output'):
        os.mkdir('output')
    if not os.path.isdir('output/test_s' + str(args.checksession)):
        os.mkdir('output/test_s' + str(args.checksession))

    pickle.dump(meanAP_epoch[:, IoUidx, :], open(
            'output/test_s' + str(args.checksession) + '/meanAP_s' + str(args.checksession) + '_epoch_' + str(
                epoch + 1) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    mat = np.matrix(meanAP_epoch[:, IoUidx, :])
    with open('output/test_s' + str(args.checksession) + '/meanAP_s' + str(args.checksession) + '_IoU_' + str(
            IoU)[2:] + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')
    mat = np.matrix(recall[:, IoUidx, :])
    with open('output/test_s' + str(args.checksession) + '/recall_s' + str(args.checksession) + '_IoU_' + str(
            IoU)[2:] + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')
    mat = np.matrix(precision[:, IoUidx, :])
    with open('output/test_s' + str(args.checksession) + '/precision_s' + str(
            args.checksession) + '_IoU_' + str(IoU)[2:] + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')
    mat = np.matrix(meanTime[:, IoUidx, :])
    with open('output/test_s' + str(args.checksession) + '/meanTime_s' + str(args.checksession) + '_IoU_' + str(
            IoU)[2:] + '.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')
  #pickle.dump(meanAP_epoch,open('output/meanAP_s'+str(args.checksession)+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)
  #pickle.dump(recall, open('output/recall_s' + str(args.checksession) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
  #pickle.dump(precision, open('output/precision_s' + str(args.checksession) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

