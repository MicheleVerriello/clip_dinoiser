import os
from models.builder import build_model
from hydra import compose, initialize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
import torch
from segmentation.datasets.pascal_context import PascalContextDataset
import argparse
from helpers.visualization import mask2rgb
from typing import List

initialize(config_path="configs", version_base=None)
PALETTE = list(PascalContextDataset.PALETTE)


def visualize_per_image(file_path: str, TEXT_PROMPTS: List[str], model: torch.nn.Module, device: str, output_dir: str,
                        ):
    """
    Visualizes output segmentation mask and saves it alongside with the labels in a file in a given output directory.

    :param file_path: [str] path to the image file
    :param TEXT_PROMPTS: [list(str)] list of text prompts to use for segmentation
    :param model: [torch.nn.module] loaded model for inference
    :param device: either "cpu" or "cuda"
    :param output_dir: [str] output directory
    :return:
    """
    assert os.path.isfile(file_path), f"No such file: {file_path}"

    img = Image.open(file_path).convert('RGB')
    img_tens = T.PILToTensor()(img).unsqueeze(0).to(device) / 255.

    h, w = img_tens.shape[-2:]
    name = file_path.split('.')[0]

    output = model(img_tens).cpu()
    output = F.interpolate(output, scale_factor=model.vit_patch_size, mode="bilinear",
                           align_corners=False)[..., :h, :w]
    output = output[0].argmax(dim=0)
    mask = mask2rgb(output, PALETTE)
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{name}_ours.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # save labels in a separate file
    fig = plt.figure()
    classes = np.unique(output).tolist()
    plt.imshow(np.array(itemgetter(*classes)(PALETTE)).reshape(1, -1, 3))
    plt.xticks(np.arange(len(classes)), list(itemgetter(*classes)(TEXT_PROMPTS)), rotation=45)
    plt.yticks([])
    plt.savefig(os.path.join(output_dir, f'{name}_labels.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for CLIP-DINOiser')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--support_images', type=str, nargs='+', required=True, help='Paths to the support images')
    args = parser.parse_args()
    return args

def list_of_strings(arg):
    return arg.split(',')

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return transform(image).unsqueeze(0)

def load_support_images(support_image_paths):
    support_images = [load_image(image_path) for image_path in support_image_paths]
    return torch.cat(support_images, dim=0)


def main():
    args = parse_args()
    
    # Load the input image
    input_image = load_image(args.file_path)
    
    # Load the support images
    support_images = load_support_images(args.support_images)
    
    # Initialize the model
    model = build_model()
    
    # Generate embeddings for the support images
    with torch.no_grad():
        support_embeddings = model.clip_backbone.encode_image(support_images)
    
    # Perform inference using the support embeddings
    h, w = input_image.shape[-2:]
    output = model(input_image, support_embeddings).cpu()
    output = F.interpolate(output, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False)[..., :h, :w]
    output = output[0].argmax(dim=0)
    mask = mask2rgb(output, PALETTE)
    
    # Visualize the results
    fig = plt.figure(figsize=(3, 1))
    classes = np.unique(output).tolist()
    plt.imshow(np.array(itemgetter(*classes)(PALETTE)).reshape(1, -1, 3))
    plt.xticks(np.arange(len(classes)), [f"Class {i}" for i in classes], rotation=45)
    plt.yticks([])
    plt.savefig(os.path.join(output_dir, f'{name}_labels.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return mask, fig, input_image

if __name__ == '__main__':
    main()
