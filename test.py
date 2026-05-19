import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask_true
from utils.dice_score import dice_coeff
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import predict

if __name__ == '__main__':
    model_file = "checkpoints/checkpoint_epoch5.pth"
    no_save = False
    viz = False
    imgs_dir = "test_data/imgs"
    masks_dir = "test_data/masks"
    out_dir = "test_data/out"
    dice_score = 0
    num_val_batches = 0
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # fig = plt.figure(figsize=[10, 10])

    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_file}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model_file, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')
    
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

        mask_pred, mask_pred_tensor = predict.predict_img(net=net,
                        full_img=img,
                        scale_factor=1.0,
                        out_threshold=0.3,
                        device=device)

        # move images and labels to correct device and type
        try:
            mask_true = Image.open(mask_filepath)
        except:
            mask_true = Image.open(mask_gif_filepath)
            
        mask_true_tensor = transform(mask_true)
        dice_score_local = dice_coeff(mask_pred_tensor, mask_true_tensor, reduce_batch_first=False)
        print("Dice score: " + str(dice_score_local))
        dice_score += dice_score_local
        num_val_batches += 1

        # cm = confusion_matrix(mask_true, mask_pred)

        # ax = fig.add_subplot(1, 2, 2)
        # c = ConfusionMatrixDisplay(cm, display_labels=range(len(np.unique(mask_true))))
        # c.plot(ax = ax)
        
        # print(classification_report(mask_true, mask_pred))
            
        if not no_save:
            result = predict.mask_to_image(mask_pred, mask_values)
            result.save(out_filepath)
            logging.info(f'Mask saved to {out_filepath}')

        if viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask_true(img, mask_pred, mask_true)

    dice_score /= max(num_val_batches, 1)
    print("Mean Dice score: " + str(dice_score))
