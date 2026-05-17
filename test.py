import logging
import os

import torch
from PIL import Image
import numpy as np

from unet import UNet
from utils.utils import plot_img_and_mask

import predict

if __name__ == '__main__':
    model_file = "checkpoints/checkpoint_epoch5.pth"
    no_save = True
    viz = True
    imgs_dir = "test_data/imgs"
    out_dir = "test_data/out"
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    for file in os.listdir(imgs_dir):
        filename = os.fsdecode(file)

        filepath = os.path.join(imgs_dir, filename)
        out_filepath = os.path.join(out_dir, filename)

        net = UNet(n_channels=3, n_classes=2, bilinear=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Loading model {model_file}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        state_dict = torch.load(model_file, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')

        logging.info(f'Predicting image {filepath} ...')
        img = Image.open(filepath)

        mask = predict.predict_img(net=net,
                        full_img=np.array(img),
                        scale_factor=1.0,
                        out_threshold=0.5,
                        device=device)

        if not no_save:
            result = predict.mask_to_image(mask, mask_values)
            result.save(out_filepath)
            logging.info(f'Mask saved to {out_filepath}')

        if viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
