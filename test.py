import logging
import os

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from unet import UNet
from utils.utils import plot_img_and_mask_true
from utils.dice_score import dice_coeff
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, classification_report
import matplotlib.pyplot as plt
import cv2
import numpy as np

import predict

def model_load(model_file):
    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_file}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model_file, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    return net, device, mask_values

def pr_curve():
    model_file = "checkpoints/checkpoint_epoch5.pth"
    imgs_dir = "test_data/archive_correct/imgs"
    masks_dir = "test_data/archive_correct/masks"
    # out_dir = "test_data/archive_correct/out"

    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net, device, mask_values = model_load(model_file)

    all_scores = []
    all_masks = []
    for file in os.listdir(imgs_dir):
        filename = os.fsdecode(file)
        filename_mask = Path(filename).stem+"_mask.png"
        filename_mask_gif = Path(filename).stem+"_mask.gif"
        # filename_out = Path(filename).stem+".png"

        filepath = os.path.join(imgs_dir, filename)
        mask_filepath = os.path.join(masks_dir, filename_mask)
        mask_gif_filepath = os.path.join(masks_dir, filename_mask_gif)
        # out_filepath = os.path.join(out_dir, filename_out)

        logging.info(f'Predicting image {filepath} ...')
        logging.info(f'True mask image {mask_filepath} ...')
        img = Image.open(filepath)

        # for th in thresholds:
        # out_filepath = "test_data/out/"+file+"_score"+".png"
        
        _, score = predict.predict_img(net=net,
                        full_img=img,
                        scale_factor=1.0,
                        out_threshold=0.5,
                        device=device)
        
        true_mask = cv2.imread(mask_filepath)
        true_mask = cv2.cvtColor(true_mask, cv2.COLOR_BGR2GRAY)
        
        all_scores.append(score.flatten())
        all_masks.append(true_mask.flatten())
        
    all_scores = np.concatenate(all_scores)
    all_masks = np.concatenate(all_masks)

    prec, recall, _ = precision_recall_curve(all_masks, all_scores)

    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.show()
    

def model_test():
    model_file = "checkpoints/checkpoint_epoch5.pth"
    no_save = False
    viz = False
    imgs_dir = "test_data/imgs"
    masks_dir = "test_data/masks"
    out_dir = "test_data/out"
    dice_score = 0
    num_val_batches = 0
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net, device, mask_values = model_load(model_file)
    
    for file in os.listdir(imgs_dir):
        filename = os.fsdecode(file)
        filename_mask = Path(filename).stem+"_mask.png"
        filename_mask_gif = Path(filename).stem+"_mask.gif"
        filename_out = Path(filename).stem+".png"

        filepath = os.path.join(imgs_dir, filename)
        mask_filepath = os.path.join(masks_dir, filename_mask)
        mask_gif_filepath = os.path.join(masks_dir, filename_mask_gif)
        out_filepath = os.path.join(out_dir, filename_out)

        logging.info(f'Predicting image {filepath} ...')
        logging.info(f'True mask image {mask_filepath} ...')
        img = Image.open(filepath)

        mask_pred = predict.predict_img(net=net,
                        full_img=img,
                        scale_factor=1.0,
                        out_threshold=0.3,
                        device=device)

        # move images and labels to correct device and type
        try:
            mask_true = Image.open(mask_filepath)
        except:
            mask_true = Image.open(mask_gif_filepath)
            
        if not no_save:
            result = predict.mask_to_image(mask_pred, mask_values)
            result.save(out_filepath)
            logging.info(f'Mask saved to {out_filepath}')

        if viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask_true(img, mask_pred, mask_true)

    dice_score /= max(num_val_batches, 1)
    print("Mean Dice score: " + str(dice_score))


if __name__ == '__main__':
    # model_test()
    pr_curve()