import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
# from tqdm import notebook
from tqdm import tqdm 
from torchvision.utils import save_image, make_grid
import random
from torch.cuda.amp import autocast, GradScaler # added due to cuda memory 

manualSeed = 123
random.seed(manualSeed)
torch.manual_seed(manualSeed)

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import glob

DATA_DIR = "C:/Users/User/Desktop/Cerebra/Week_1/home/ubuntu/zhuldyzzhan/tissue_segmentation/src/notebooks/norm_images_new_fov"

stats = (0.5, 0.5)

def denorm(img_tensors):
    return img_tensors * stats[0] + stats[1]

config = {
    "IMAGE_SIZE": 256,
    "BATCH_SIZE": 8,    # changed due to cuda memory
    "LR": 1e-4,
    "CRITIC_ITERATIONS": 5,
    "MAX_EPOCHS": 1000,
    "LAMBDA_GP": 10,
    "MAX_PIXEL_VAL": 50,
    "BETAS": (0.0, 0.9),
    "OPTIMIZATION": "adam"
}

# Create GradScaler objects
scaler_d = GradScaler()
scaler_g = GradScaler()


class Dataset:
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tissuemask_name = file[:-4]+"_mask.png"
        tissuemask = cv2.imread(tissuemask_name, 0)
#         print(tissuemask_name, os.path.exists(tissuemask_name))
        tissuemask = (tissuemask > 0).astype(np.uint8)
        real_image = image.copy()
        image[:, 128:] = np.fliplr(image[:,:128].copy())
        tissuemask[:,:128] = np.fliplr(tissuemask[:,128:].copy())
        real_image[:,:128] = np.fliplr(real_image[:,128:].copy())
        tissuemask = tissuemask*80
        if self.transform:
            sample = self.transform(image=image, real_image=real_image, tissuemask=tissuemask)
            image, real_image, tissuemask = sample["image"], sample["real_image"], sample["tissuemask"]
            
        return image, tissuemask, real_image

train_transform = albu.Compose([
    albu.Resize(height=config['IMAGE_SIZE'], width=config['IMAGE_SIZE']),
    albu.CenterCrop(height=config['IMAGE_SIZE'], width=config['IMAGE_SIZE']),
    
    albu.Normalize(mean=stats[0], std=stats[1], max_pixel_value=config['MAX_PIXEL_VAL']),
    ToTensor()
], additional_targets={"real_image": "image", "tissuemask": "image", "image": "image"})


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

latent_size = 128


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #64x256x256
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #128, 128, 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #256, 64, 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #512, 32, 32
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #1024, 16, 16
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        #512, 32, 32
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        #256, 64, 64
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        #128, 128, 128
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, condition):
        x = torch.cat([x,condition], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        upconv1 = self.upconv1(conv5)
        concat4_1 = torch.cat([conv4, upconv1], dim=1)
        upconv2 = self.upconv2(concat4_1)
        
        concat3_2 = torch.cat([conv3, upconv2], dim=1)
        upconv3 = self.upconv3(concat3_2)
        
        concat2_3 = torch.cat([conv2, upconv3], dim=1)
        upconv4 = self.upconv4(concat2_3)
        
        concat1_4 = torch.cat([conv1, upconv4], dim=1)
        upconv5 = self.upconv5(concat1_4)
        out = self.output(upconv5)
        return out
    
generator = Generator()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.downsampling = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
#             nn.Sigmoid()
        )
        
    def forward(self, x, condition):
        x = torch.cat([x,condition], dim=1)
        return self.downsampling(x)
    
critic = Discriminator()

device = torch.device("cuda")
critic = critic.to(device)
critic.apply(weights_init)
generator = generator.to(device)
generator.apply(weights_init)

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, masked_images, condition, show=True):
    fake_images = generator(masked_images, condition)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    grid = denorm(fake_images).cpu().detach()
#     save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        grid = make_grid(fake_images.cpu().detach(), nrow=8)
        ax.imshow(grid.permute(1, 2, 0))
    return grid

smoothing = 0.1


def gradient_penalty(critic, real, fake, condition, device="cuda"):
    B, C, H, W = real.shape
    alpha = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(device)
#     print(real.shape, fake.shape, alpha.shape)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, condition)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

l1_loss = nn.L1Loss(reduction='none')


def train_discriminator(masked_images, real_images, tissuemask, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()
    B, C, H, W = real_images.shape
    # Generate fake images
#     latent = torch.randn(B, latent_size, 1, 1, device=device)
    fake_images = generator(masked_images, tissuemask)
    # Pass fake images through discriminator
    fake_preds = critic(fake_images, tissuemask).reshape(-1)
    real_preds = critic(real_images, tissuemask).reshape(-1)
    
    gp = gradient_penalty(critic, real_images, fake_images, tissuemask, device=device)
    
    critic_loss = - (torch.mean(real_preds) - torch.mean(fake_preds)) + config['LAMBDA_GP'] * gp
    
    # L1 loss on masked region
    mask = tissuemask.clone()
    mask[mask > 1] = 1
#     print(mask.shape, masked_images[...,128:].sum(), masked_images[...,:128].sum())
    nonzero_n = mask.sum()
    loss_l1 = (l1_loss(fake_images, real_images) * mask.float()).sum() / nonzero_n

    full_loss = critic_loss + loss_l1
#     full_loss = critic_loss

    # Scale the loss before calling backward
    scaler_d.scale(full_loss).backward(retain_graph=True)

    # Update discriminator weights
    scaler_d.step(opt_d)
    scaler_d.update()

    # full_loss.backward(retain_graph=True)
    # opt_d.step()
    
    return full_loss.item(), loss_l1.item()


def train_generator(masked_images, real_images, tissuemask, opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
#     latent = torch.randn(BATCH_SIZE, latent_size, 1, 1, device=device)
    fake_images = generator(masked_images, tissuemask)
    
    # Try to fool the discriminator
    preds = critic(fake_images, tissuemask).reshape(-1)
    gen_loss = -torch.mean(preds)

    # L1 loss on masked region
    mask = tissuemask.clone()
    mask[mask > 1] = 1
#     print(mask.shape, masked_images[...,128:].sum(), masked_images[...,:128].sum())
    nonzero_n = mask.sum()
    loss_l1 = (l1_loss(fake_images, real_images) * mask.float()).sum() / nonzero_n

    full_loss = loss_l1 + gen_loss
#     full_loss = gen_loss

    # Scale the loss before calling backward
    scaler_g.scale(full_loss).backward()

    # Update generator weights
    scaler_g.step(opt_g)
    scaler_g.update()

    # Update generator weights
    # full_loss.backward()
    # opt_g.step()
    
    return full_loss.item(), loss_l1.item()


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    
    for epoch in range(epochs):
        for masked_images, tissuemask, real_images in tqdm(train_dl, leave=False):
            masked_images = masked_images.to(device)
            tissuemask = tissuemask.to(device)
            real_images = real_images.to(device)
            # Train discriminator: maximize 
            for _ in range(config['CRITIC_ITERATIONS']):
                loss_d, lossL1_d = train_discriminator(masked_images, real_images, tissuemask, opt_d)
            # Train generator: minimize
            loss_g, lossL1_g = train_generator(masked_images, real_images, tissuemask, opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d))
        
        img_list = []
        # Save generated images
        fake = save_samples(epoch+start_idx, masked_images, tissuemask, show=False)
        img_list.append(make_grid(fake, padding=2, normalize=True))
        
        masked_img_list = []
        masked_img_list.append(make_grid(masked_images, padding=2, normalize=True))
        
        if (epoch+1) % 50 == 0:
            torch.save(generator.state_dict(), f"model_epoch{str(epoch+1)}.h5")
    return losses_g, losses_d


if __name__ == "__main__":
    files = glob.glob(DATA_DIR+"/**/*.png")
    files = [f for f in files if not os.path.basename(f)[:-4].endswith("mask")]
    print(len(files))

    train_ds = Dataset(files, transform=train_transform)
    train_dl = DataLoader(train_ds, config['BATCH_SIZE'], shuffle=True, num_workers=3, pin_memory=True)

    x, tissuemask, real_image  = train_ds[50]

    real_image.shape

    x.min(), x.max(), real_image.min(), real_image.max(), tissuemask.min(), tissuemask.max()

    vis = (x * 0.5 + 0.5).cpu().detach().squeeze().numpy() * 50

    (vis).mean()

    MAX_EPOCHS = 1000
    history = fit(MAX_EPOCHS, config['LR'])

