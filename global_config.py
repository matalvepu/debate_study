import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "driver.py"

DEVICE = torch.device("cuda:0")
MOSI_ACOUSTIC_DIM = 74
MOSI_VISUAL_DIM = 47

HUMOR_ACOUSTIC_DIM = 81
#HUMOR_VISUAL_DIM = 371
HUMOR_VISUAL_DIM = 91

ACOUSTIC_DIM = 81
VISUAL_DIM = 91

H_MERGE_SENT = 768
DATASET_LOCATION = "/scratch/mhasan8/processed_multimodal_data/"
