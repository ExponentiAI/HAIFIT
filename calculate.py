import torch
import torchvision.transforms.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
from lpips import LPIPS
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms


preprocess = transforms.Compose([
    transforms.Resize((224, 224))
])

def calculate_ssim(image1, image2):
    return ssim(image1, image2, channel_axis=-1, data_range=1.0)

def calculate_psnr(image1, image2):
    return psnr(image1, image2)

def calculate_lpips(image1, image2, lpips, device):
    image1_tensor = F.to_tensor(image1).unsqueeze(0).to(device)
    image2_tensor = F.to_tensor(image2).unsqueeze(0).to(device)
    distance = lpips(image1_tensor, image2_tensor)
    return distance.item()

def calculate_fid(folder1, folder2, device):
    return fid_score.calculate_fid_given_paths([folder1, folder2], batch_size=32, device=device, dims=2048)

def calculate_metrics(folder1, folder2):
    image_list1 = os.listdir(folder1)
    image_list2 = os.listdir(folder2)

    metrics = {
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'FID': []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = LPIPS(net="squeeze").to(device)

    for image_name1, image_name2 in tqdm(zip(image_list1, image_list2)):
        image_path1 = os.path.join(folder1, image_name1)
        image_path2 = os.path.join(folder2, image_name2)
        image1 = np.array(preprocess(Image.open(image_path1).convert("RGB")))
        image2 = np.array(preprocess(Image.open(image_path2).convert("RGB")))

        ssim_value = calculate_ssim(image1, image2)
        psnr_value = calculate_psnr(image1, image2)
        lpips_value = calculate_lpips(image1, image2, lpips, device)

        metrics['SSIM'].append(ssim_value)
        metrics['PSNR'].append(psnr_value)
        metrics['LPIPS'].append(lpips_value)

    fid_value = calculate_fid(folder1, folder2, device)

    mean_metrics = {
        'SSIM': sum(metrics['SSIM']) / len(metrics['SSIM']),
        'PSNR': sum(metrics['PSNR']) / len(metrics['PSNR']),
        'LPIPS': sum(metrics['LPIPS']) / len(metrics['LPIPS']),
        'FID': fid_value
    }

    return mean_metrics


if __name__ == '__main__':
    folder1 = './HAIGEN/generate/lstm/gen/'
    folder2 = './HAIGEN/generate/lstm/org/'

    mean_metrics = calculate_metrics(folder1, folder2)

    print("Mean Metrics:")
    for metric, value in mean_metrics.items():
        print(metric + ": " + str(value))
