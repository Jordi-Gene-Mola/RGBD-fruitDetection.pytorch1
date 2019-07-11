# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import xml.etree.ElementTree as ET
import os
import numpy as np
import time

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    '''obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)'''
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects

#Normalment esta a false el use_07..
def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh,
             minconfid,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile[:-4])
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]
  f.close()
  
  if not os.path.isfile(cachefile):
    # load annotations
    print(annopath)
    recs = {}
    for i, imagename in enumerate(imagenames):
      #recs[imagename] = parse_rec(annopath.format(imagename))
      recs[imagename] = parse_rec(annopath.format(imagename)[:100] + '_RGB.xml')
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
       pickle.dump(recs, f)
    f.close()
  else:
    # load
    # print('abans pickle')
    # print(cachefile)
    # f=open(cachefile, 'rb')
    # recs = pickle.load(f)
    # f.close()
    # print('despres pickle')
    # print('abans pickle')
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        # print('abans pickle 2')
        recs = pickle.load(f, encoding='bytes')
        # print('despres pickle 2')
    # print('despres pickle')
    # time.sleep(1)
    f.close()
    del(f)
  print('estem aqui')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}
  print('estem aqui 2')
  # read dets
  # print(detpath)
  detfile = detpath.format(classname)
  print(detfile)
  with open(detfile, 'r') as f:
    lines = f.readlines()
  f.close()
  print('estem aqui 3')

  splitlines = [x.strip().split(' ') for x in lines]
  print('estem aqui 4')
  #image_ids = [x[0]+' '+x[1]+' '+x[2]+' '+x[3] for x in splitlines]
  #confidence = np.array([float(x[4]) for x in splitlines])
  #BB = np.array([[float(z) for z in x[5:]] for x in splitlines])
  image_ids = [x[0] for x in splitlines]
  print('estem aqui 5')
  confidence = np.array([float(x[1]) for x in splitlines])
  print('estem aqui 6')
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  print('estem aqui 7')
  # BB = np.zeros([len(splitlines), 4])
  # lineIdx = 0
  # for x in splitlines:
  #     #print(i)
  #     #print(x[2:])
  #     #print(lineIdx)
  #     BB[lineIdx, :] = x[2:]
  #     lineIdx = lineIdx + 1



  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    print('estem aqui 8')
    sorted_ind = np.argsort(-confidence)
    print('estem aqui 9')
    sorted_scores = np.sort(-confidence)
    print('estem aqui 10')
    BB = BB[sorted_ind, :]
    print('estem aqui 11')
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  print('estem aqui 12')
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  
  rec = tp / float(npos)
  r=rec[sum(confidence>minconfid)]
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  p = prec[sum(confidence > minconfid)]
  ap = voc_ap(rec, prec, use_07_metric)


  return rec, prec, ap, r, p
