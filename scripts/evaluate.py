import numpy as np
import pandas as pd
import cv2
import torch
import shutil
import blobfile as bf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchmetrics import JaccardIndex, Dice

def visualize(img):
    '''
    Normalize for visualization.
    '''
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def diff(org, sample):
    return abs(visualize(org) - sample)

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()

def visualize(img):
    '''
    Normalize for visualization.
    '''
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def heatmap2segmentationmap(difference, threshold=0.2):
    output = []
    for row in difference:
        tmp = []
        for col in row: 
            if col < threshold:
                tmp.append(0)
            else:
                tmp.append(1)
        output.append(tmp)
    return output

def iou(path, jaccard_threshold, heatmap_threshold):
    gt_path = "/home/pill/lung/Pytorch-UNet/test_data/imgs/" + path
    mask_path = "/home/pill/lung/Pytorch-UNet/test_data/masks/" + path
    gt = np.load(gt_path)
    gt = cv2.resize(gt, (256, 256))
    mask = np.load(mask_path).astype('int16')
    mask = cv2.resize(mask, (256, 256))

    sample_path = "/home/pill/lung/diffusion-anomaly-detection/sample_results/" + path
    # sample_path = "/home/pill/lung/Pytorch-UNet/results/sample_results/" + path #Unet
    sample = np.load(sample_path)
    sample = sample.astype('int16')
    sample = cv2.resize(sample, (256, 256))
    difference = diff(gt, sample)

    pred = heatmap2segmentationmap(difference, heatmap_threshold)

    # jaccard = JaccardIndex(task="binary", ignore_index=0, threshold=jaccard_threshold)
    jaccard = Dice(average='micro') #, ignore_index=0)
    return jaccard(torch.tensor(mask, dtype=torch.int8), torch.tensor(pred, dtype=torch.int8))
    # print(jaccard(torch.tensor([[1, 0], [0, 1]], dtype=torch.int8), torch.tensor([[1, 1], [1, 0]], dtype=torch.int8)))
    # print(torch.tensor(mask, dtype=torch.int8))
    # print(torch.tensor(difference, dtype=torch.int8))

if __name__ == "__main__":

    heatmap_threshold = 0.2
    jaccard_threshold = 0.5


    # files = pd.read_csv("/home/pill/lung/diffusion-anomaly-detection/results/test.csv", header=None)
    files = pd.read_csv("/home/pill/lung/diffusion-anomaly-detection/results/test3.csv", header=None)
    output = []
    # gts = []
    # preds = []
    for i in range(files.__len__()):
        tmp = iou(files.loc[i][0].split('/')[-1], jaccard_threshold, heatmap_threshold)
        output.append(tmp)
        # gt, pred = get_mask_and_difference(files.loc[i][0].split('/')[-1])
        # gts.append(gt)
        # preds.append(pred)
    # output = iou2(torch.tensor(gts, dtype=torch.int8), torch.tensor(preds, dtype=torch.int8), n_classes=2)
    print(len(output))
    # print(output)
    sum = 0
    
    cnt = 0
    tmp = 0
    for o in output:
        if not torch.isnan(o):
            sum = sum + o
            cnt = cnt + 1
        else: 
            tmp = tmp + 1 # full black
    print(float(sum) / float(cnt))
    print(tmp)
# Jaccard
# 0.2 - 0.5 - 0.0023182621996255186 - 995
# 0.15 - 0.5 - 0.0008367489235035116 - 602
# 0.2 - 0.2 - 0.0023182621996255186 - 995
# 0.5 - 0.2 - 0.0023182621996255186 - 995

# 3.577673388092885e-05 - Unet

# Dice
