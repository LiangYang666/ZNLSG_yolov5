import os
import json
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle




#%%
def draw_bbox(ann,category_info , c='r', dist = 10):

    ax = plt.gca()
    ax.set_autoscale_on(False)
    category_id = ann['category_id']
    print(category_id)
    category_name = category_info['name']

    # c = list(self.colors[category_id % len(self.colors)])
    # c = [i / 255 for i in c]
    [bbox_x1, bbox_y1, bbox_w, bbox_h] = ann['bbox']
    ax.add_patch(
        plt.Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h, color=c, fill=False, linewidth=2))
    ax.text(bbox_x1, bbox_y1+dist, str(category_id)+' '+category_name, fontsize=10, color='white',
            bbox={'facecolor': c, 'alpha': 0.5})

def draw_bboxes(coco: COCO, imageid: int, c='r', dist=10):
    annotationIds = coco.getAnnIds(imageid)
    anns = coco.loadAnns(annotationIds)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in anns:
        category_id = ann['category_id']
        category_info = coco.loadCats(category_id)[0]
        category_name = category_info['name']
        # c = list(self.colors[category_id % len(self.colors)])
        # c = [i / 255 for i in c]
        [bbox_x1, bbox_y1, bbox_w, bbox_h] = ann['bbox']
        ax.add_patch(
            plt.Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h, color=c, fill=False, linewidth=2))
        ax.text(bbox_x1, bbox_y1 + dist, category_name, fontsize=10, color='white',
                bbox={'facecolor': c, 'alpha': 0.5})
        # ax.text(bbox_x, bbox_y-3, category_name, fontsize=16, color=c)

def on_key(event):
    print(event.key)
def computeIou(a, b):
    """
    :param a: bbox like coco annotation, [x1, y1, w, h]
    :param b: bbox like coco annotation, [x1, y1, w, h]
    :return : IOU
    """
    aBox_xyxy = [a[0], a[1], a[0]+a[2], a[1]+a[3]]
    bBox_xyxy = [b[0], b[1], b[0]+b[2], b[1]+b[3]]

    width0 = aBox_xyxy[2] - aBox_xyxy[0]
    height0 = aBox_xyxy[3] - aBox_xyxy[1]
    width1 = bBox_xyxy[2] - bBox_xyxy[0]
    height1 = bBox_xyxy[3] - bBox_xyxy[1]
    max_x = max(aBox_xyxy[2], bBox_xyxy[2])
    min_x = min(aBox_xyxy[0], bBox_xyxy[0])
    width = width0 + width1 - (max_x - min_x)
    max_y = max(aBox_xyxy[3], bBox_xyxy[3])
    min_y = min(aBox_xyxy[1], bBox_xyxy[1])
    height = height0 + height1 - (max_y - min_y)
    if width <= 0 or height <= 0:
        return 0
    interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou



if __name__ == '__main__':
    data_dir = "../../../data"
    true_train_a_annotations_file = os.path.join(data_dir, "cssjj/test/a_annotations.json")
    pred_train_a_annotations_file = os.path.join(data_dir, "cssjj/test/pred_a_annotations_top7.json")
    trueCoco = COCO(true_train_a_annotations_file)
    predCoco = COCO(pred_train_a_annotations_file)
    imgIds = trueCoco.getImgIds()
    fig = plt.Figure()
    # fig.canvas.mpl_connect('key_release_event', on_key)
    sorted(imgIds)
    id = 0
    while id < len(imgIds):
        imgId = imgIds[id]
        imgInfo = trueCoco.loadImgs(imgId)[0]
        imgFile = os.path.join(os.path.dirname(true_train_a_annotations_file)+'/a_images', imgInfo['file_name'])
        img = plt.imread(imgFile)
        # exit(0)
        plt.imshow(img)
        ax = plt.gca()
        ax.text(0, 0 + 30, str(id) + '  '+imgInfo['file_name'], fontsize=10, color='white',
                bbox={'facecolor': 'r', 'alpha': 0.5})
        # draw_bboxes(trueCoco, imgId, 'r', 0)
        # draw_bboxes(predCoco, imgId, 'g', 30)

        trueAnnIds = trueCoco.getAnnIds(imgIds=imgId)
        predAnnIds = predCoco.getAnnIds(imgIds=imgId)
        trueAnnIdsIouBigMask = [False] * len(trueAnnIds)
        predAnnIdsIouBigMask = [False] * len(predAnnIds)
        iouBigPairs = []
        for i, trueAnnId in enumerate(trueAnnIds):
            trueAnn = trueCoco.loadAnns(trueAnnId)[0]
            trueBox = trueAnn['bbox']
            for j, predAnnId in enumerate(predAnnIds):
                predAnn = predCoco.loadAnns(predAnnId)[0]
                predBox = predAnn['bbox']
                if computeIou(trueBox, predBox) > 0.75:
                    iouBigPairs.append([trueAnn, predAnn])
                    trueAnnIdsIouBigMask[i] = True
                    predAnnIdsIouBigMask[j] = True
        for iouBigPair in iouBigPairs:
            if iouBigPair[0]['category_id'] == iouBigPair[1]['category_id']:
                draw_bbox(iouBigPair[1], category_info=predCoco.loadCats(iouBigPair[1]['category_id'])[0], dist=0, c='w')
            else:
                draw_bbox(iouBigPair[0], category_info=trueCoco.loadCats(iouBigPair[0]['category_id'])[0], dist=0, c='b')
                draw_bbox(iouBigPair[1], category_info=predCoco.loadCats(iouBigPair[1]['category_id'])[0], dist=30, c='b')

        for i, mask in enumerate(trueAnnIdsIouBigMask):
            if mask:
                continue
            else:
                annId = trueAnnIds[i]
                trueAnn = trueCoco.loadAnns(annId)[0]
                draw_bbox(trueAnn, category_info=trueCoco.loadCats(trueAnn['category_id'])[0], dist=0, c='g')
        for i, mask in enumerate(predAnnIdsIouBigMask):
            if mask:
                continue
            else:
                annId = predAnnIds[i]
                predAnn = predCoco.loadAnns(annId)[0]
                draw_bbox(predAnn, category_info=predCoco.loadCats(predAnn['category_id'])[0], dist=30, c='r')


        plt.pause(0.01)
        key_press = 0

        while True:
            pos = plt.ginput(n=1, timeout=1000)   # n is times
            # print(pos)
            if len(pos) > 0:
                if pos[0][0] > 600:
                    id += 1
                    break
                if pos[0][0] < 300:
                    id -= 1
                    break

        # while not key_press:
        #     key_press = plt.waitforbuttonpress()
        # print(key_press)
        # plt.waitforbuttonpress()
        plt.cla()


















