#!/usr/bin/env python
# coding: utf-8




import torch
import torchvision.transforms as T
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt





import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("This is an informational message")
logging.warning("This is a warning message")

# You can downlaod the dataset using: kaggle competitions download -c imagenet-object-localization-challenge


IMAGE_SIZE = 128
CENTER_SIZE = 64
BATCH_SIZE = 1500

SELECTED_CLASSES =  {
            # 858: "Tile Roof", # Tile Roof
                        # 937: "Broccoli", 
                        # 980: "Volcano",
                        987: "Corn",
                        949: 'Strawberry',
                        948: "Apple",
                        954: "Banana",
                        953: "Pineapple",
                        951: "Lemon",
                        950: 'Orange',
                        967: 'Pomegranate'
}


# # Utils




import math
def display_image(img):
        with torch.no_grad():
                img = img.permute((1, 2, 0))
                plt.imshow(img.cpu().numpy())
                plt.show()

def display_images(imgs, images_per_row=2, title=None):
        with torch.no_grad():
                rows = math.ceil(len(imgs) / images_per_row)
                fig = plt.figure(figsize=(images_per_row*2, rows*2))
                if title:
                        fig.suptitle(title, fontsize=10)
                for i, img in enumerate(imgs):
                        plt.subplot(rows, images_per_row, i+1)
                        plt.axis('off')
                        plt.tight_layout()
                        img = img.permute((1, 2, 0))
                        plt.imshow(img.cpu().numpy())
                plt.show()

def get_center_of_image(img, size=64):
        return T.CenterCrop((size, size))(img)

def get_img_without_center(img, size=64):
        left = (IMAGE_SIZE - CENTER_SIZE) // 2
        right = (IMAGE_SIZE - CENTER_SIZE) // 2 + CENTER_SIZE
        if(len(img.shape) == 3):
                img[ :, left:right, left:right ] = 0
        elif(len(img.shape) == 4):
                img[:, :, left:right, left:right ] = 0
        else:
                raise RuntimeError("Wrong shape to get_img_without_center!")
                
        return img

class CenterImageRemoval(object):
    def __init__(self, image_size, center_size):
        self.image_size = image_size
        self.center_size = center_size
        
    def __call__(self, img):
        left = (self.image_size - self.center_size) // 2
        right = (self.image_size - self.center_size) // 2 + self.center_size

        if(len(img.shape) == 3):
                assert img.shape[1] == img.shape[2] == self.image_size
                img[ :, left:right, left:right ] = 0
        elif(len(img.shape) == 4):
                assert img.shape[2] == img.shape[3] == self.image_size
                img[:, :, left:right, left:right ] = 0

        return img

class AddImageCenter(object):
    def __init__(self, image_size, center_size):
        self.image_size = image_size
        self.center_size = center_size
        
    def __call__(self, img, center):
        assert center.shape[1] == center.shape[2] == self.center_size

        left = (self.image_size - self.center_size) // 2
        right = (self.image_size - self.center_size) // 2 + self.center_size

        if(len(img.shape) == 3):
                assert img.shape[1] == img.shape[2] == self.image_size
                img[ :, left:right, left:right ] = center
        elif(len(img.shape) == 4):
                assert img.shape[2] == img.shape[3] == self.image_size
                img[:, :, left:right, left:right ] = center
        return img

class JointRandomResizeCrop(object):
    def __init__(self, size: int, minimum_scale, maximum_scale):
        """
        params:
            size (int) : size of the center crop
        """
        self.size = size
        self.min_scale = minimum_scale
        self.max_scale = maximum_scale
        
    def __call__(self, img, target):
        scale = (self.max_scale - self.min_scale)*random.random() + self.min_scale
        _, height, width = img.shape
        new_h, new_w = int(height*scale), int(width*scale)

        target = target.reshape((1, height, width))
        resized_img = T.functional.resize(img,  [new_h, new_w])
        resized_target = T.functional.resize(target, [new_h, new_w])
        crop_size = min(self.size, new_h, new_w)
        top = random.randint(0, new_h - crop_size)
        left = random.randint(0, new_w - crop_size)
        resized_img = T.functional.crop(resized_img, top, left, crop_size, crop_size)
        resized_target = T.functional.crop(resized_target, top, left, crop_size, crop_size)

        resized_img = T.functional.resize(resized_img, size=(self.size, self.size))
        resized_target = T.functional.resize(resized_target, size=(self.size, self.size))
        resized_target = resized_target.reshape((self.size, self.size))

        return (resized_img, resized_target)
    
def save_model(encoder, decoder, epoch, optimizer, path):
    state = {
    'epoch': epoch,
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)


def load_model(encoder, decoder, optimizer, path):
    state = torch.load(path)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Dataloading




train_tranforms = T.Compose([
    T.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.RandomHorizontalFlip(p=0.5),
    T.ConvertImageDtype(torch.float32),
])

val_tranforms = T.Compose([
    T.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
])

# extra transformation not applied to ground truth images
non_gt_extra_transforms = T.Compose([
    CenterImageRemoval(IMAGE_SIZE, CENTER_SIZE),
])

sanity_transforms = T.Compose([
    T.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    T.CenterCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
])





from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from IPython.display import clear_output

class CustomDataSet(Dataset):
    def __init__(self, root_dir, classes, transform=None, non_gt_extra_transforms=None, training_split=0.8, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.non_gt_extra_transforms = non_gt_extra_transforms

        selected_class_indexes = list(classes.keys())
        dirs = np.array(sorted(os.listdir(root_dir)))[selected_class_indexes]
        # print(dirs.shape)
        # train_split = int(dirs.shape[0]*training_split)
        # dirs = dirs[0:train_split]
        # print(dirs.shape)

        self.images = []
        self.cache = {}

        self.is_train = is_train
        for i, dir in enumerate(dirs):
            imgs = sorted(os.listdir(os.path.join(self.root_dir, dir)))
            train_split = int(len(imgs)*training_split)
            if is_train:
                imgs = imgs[0:train_split]
            else:
                imgs = imgs[train_split:]
            for img in imgs:
                self.images.append((os.path.join(self.root_dir, dir, img), selected_class_indexes[i]))
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, class_label = self.images[idx]
        image = None

        if img_name not in self.cache:
            self.cache[img_name] = Image.open(img_name).convert('RGB')
        image = self.cache[img_name]


        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = None

        gt_image = torch.clone(transformed_image)

        if non_gt_extra_transforms:
            transformed_image = non_gt_extra_transforms(transformed_image)

        return transformed_image, gt_image, class_label


# Create the dataset
DATASET_PATH = '/pub2/imagenet/ILSVRC/Data/CLS-LOC/train'
# Due to the way the dataset is strucutred, I can't use the offical validation images provided
# Therefore, I'm splitting the provided training_split from the dataset, 90% for training, and 10% for validation

train_dataset = CustomDataSet(root_dir=DATASET_PATH, 
                                      transform=train_tranforms, 
                                      non_gt_extra_transforms=non_gt_extra_transforms,
                                      classes=SELECTED_CLASSES,
                                      is_train=True,
                                      training_split=0.9
                                      )

validation_dataset = CustomDataSet(root_dir=DATASET_PATH, 
                                      transform=val_tranforms, 
                                      non_gt_extra_transforms=non_gt_extra_transforms,
                                      classes=SELECTED_CLASSES,
                                      is_train=False
                                      )

# Like the training dataset but only has 1 image
sanity_dataset = CustomDataSet(root_dir=DATASET_PATH, 
                                      transform=sanity_transforms, 
                                      non_gt_extra_transforms=non_gt_extra_transforms,
                                      classes=SELECTED_CLASSES,
                                      is_train=True,
                                      training_split=0.9
                                      )
sanity_dataset = torch.utils.data.Subset(sanity_dataset, [1])





train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
sanity_loader = DataLoader(sanity_dataset, batch_size=1, shuffle=True)





sample_batch = next(iter(train_loader))
sample_cutout, sample_gt, sample_class_idx = sample_batch

def display_sample_batch():
    images = []
    for i in range(10):
        images += [sample_cutout[i], sample_gt[i]]
    display_images(images)


# # Network




LATENT_SPACE_DIM = 4000
HIDDEN_LAYER_SIZE = 64

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # self.conv_t1 = nn.ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2), padding=0, output_padding=0 )
        # self.conv_1 = nn.Conv2d(128, 90, kernel_size=5, padding=1)
        # self.batch_norm_1 = nn.BatchNorm2d(90)
        # self.conv_2 = nn.Conv2d(90, 128, kernel_size=5, padding=1)
        # self.batch_norm_2 = nn.BatchNorm2d(128)
        # self.conv_t2 = nn.ConvTranspose2d(128, 3, kernel_size=5, stride=4, padding=0, output_padding=0 )
        # self.conv_final = nn.Conv2d(3, 3, 1)
        # self.num_classes = 5

        # (128, 128) -> (64, 64)
        self.conv1 = nn.Conv2d(3, HIDDEN_LAYER_SIZE, kernel_size=4, padding=1, stride=2)  

        # (64, 64) -> (32, 32)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, kernel_size=4, padding=1, stride=2)

        # (32, 32) -> (16, 16)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_SIZE, 2 * HIDDEN_LAYER_SIZE, kernel_size=4, padding=1, stride=2)  # 16x16 => 8x8

        # (16, 16) -> (8, 8)
        self.conv4 = nn.Conv2d(2 * HIDDEN_LAYER_SIZE, 4 * HIDDEN_LAYER_SIZE, kernel_size=4, padding=1, stride=2)

        # (8, 8) -> (4, 4)
        self.conv5 = nn.Conv2d(4 * HIDDEN_LAYER_SIZE, 8*HIDDEN_LAYER_SIZE, kernel_size=4, padding=1, stride=2)  # 8x8 => 4x4
        self.flatten = nn.Flatten()

        # (4, 4) -> (Latence_Space_Dim)
        self.linear = nn.Linear(4*4 * (8* HIDDEN_LAYER_SIZE), LATENT_SPACE_DIM)

    def forward(self, inp):
        # print(inp.shape)
        inp = F.leaky_relu(self.conv1(inp))
        # print(inp.shape)
        inp = F.leaky_relu(self.conv2(inp))
        # print(inp.shape)
        inp = F.leaky_relu(self.conv3(inp))
        # print(inp.shape)
        inp = F.leaky_relu(self.conv4(inp))
        # print(inp.shape)
        inp = F.leaky_relu(self.conv5(inp))
        inp = self.flatten(inp)

        # print(inp.shape)
        inp = F.leaky_relu(self.linear(inp))
        # Encoder
        # original_inp = inp
        # inp = self.resnet.conv1(inp)
        # inp = self.resnet.bn1(inp)
        # inp = self.resnet.relu(inp)
        # skip = inp.clone()
        # inp = self.resnet.maxpool(inp)
        # inp = self.resnet.layer1(inp)
        # inp = self.resnet.layer2(inp)
        # inp = self.resnet.layer3(inp)
        # inp = self.resnet.layer4(inp)
        # print("fully encoded shape is", inp.shape)
        return inp





# encoder = Encoder()
# encoder = encoder.to(device)
# encoder(sample_cutout[0].to(device).unsqueeze(0)).shape





class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # (LATENT_SPACE_DIM) -> (4 , 4)
        self.linear = nn.Linear(LATENT_SPACE_DIM, 4*4*8*HIDDEN_LAYER_SIZE)

        # (4, 4) -> (8 , 8)
        self.conv_t1 = nn.ConvTranspose2d(8*HIDDEN_LAYER_SIZE, 4*HIDDEN_LAYER_SIZE, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(4*HIDDEN_LAYER_SIZE)

        # (8, 8) -> (16, 16)
        self.conv_t2 = nn.ConvTranspose2d(4*HIDDEN_LAYER_SIZE, 2*HIDDEN_LAYER_SIZE , kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(2*HIDDEN_LAYER_SIZE)

        # (16, 16) -> (32, 32)
        self.conv_t3 = nn.ConvTranspose2d(2*HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(HIDDEN_LAYER_SIZE)

        # (32, 32) -> (64, 64)
        self.conv_t4 = nn.ConvTranspose2d(HIDDEN_LAYER_SIZE, 3, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        # self.batch_norm_4 = nn.BatchNorm2d(3)

        # self.conv = nn.Conv2d(HIDDEN_LAYER_SIZE, 3, kernel_size=4, stride=1, padding=1)
        # self.conv_t5 = nn.ConvTranspose2d(LATENT_SPACE_DIM // 64, 32, kernel_size=(2, 2), stride=(2, 2), padding=0, output_padding=1)
        # self.batch_norm_5 = nn.BatchNorm2d(32)

        # nn.Upsample(scale_factor = 2, mode='bilinear'),
        #                   nn.ReflectionPad2d(1),
        #                   nn.Conv2d(ngf * mult, int(ngf * mult / 2),
        #                                      kernel_size=3, stride=1, padding=0)



    def forward(self, inp):
        inp = self.linear(inp)
        # Reshape back to image size
        # inp.shape[0] gives the batch size
        # We want a 4x4 image shape 
        inp = inp.reshape(inp.shape[0], -1, 4, 4) 
        # print(inp.shape)

        inp = self.conv_t1(inp)
        inp = self.batch_norm_1(inp)
        inp = F.relu(inp)
        # print(inp.shape)

        inp = self.conv_t2(inp)
        inp = self.batch_norm_2(inp)
        inp = F.relu(inp)
        # print(inp.shape)

        inp = self.conv_t3(inp)
        inp = self.batch_norm_3(inp)
        inp = F.relu(inp)
        # print("here 1", inp.shape)

        inp = self.conv_t4(inp)
        # inp = self.batch_norm_4(inp)
        # inp = F.relu(inp)
        # print("here 2", inp.shape)

        # inp = self.conv_t5(inp)
        # inp = self.batch_norm_5(inp)
        # inp = F.relu(inp)
        # print(inp.shape)

        # inp = self.conv(inp)
        inp = torch.sigmoid(inp)
        
        # print("image size after decode is", inp.shape)
        # inp = T.Resize((IMAGE_SIZE, IMAGE_SIZE))(inp)

        return inp





def predict_fill_area_for_img(img, encoder, decoder):
    img = img.to(device)
    img = img.unsqueeze(0)
    embedding = encoder(img)
    out = decoder(embedding)[0]
    return out.cpu()


# # Train  




import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)





from tqdm import tqdm
from pytorch_ssim import pytorch_ssim





def get_optimizer(encoder, decoder):
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    return optim

ssim_loss = pytorch_ssim.SSIM()
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

SSIM_LOSS_RATIO = 0.1

def compute_ssim_loss(output, target):
    ssim_loss = (pytorch_ssim.ssim(output, target) + 1) / 2
    return ssim_loss

def l2andssim_loss(output, target):
    # SSIM is between -1 and 1, add 1 and divide by 2 to get rid of negativies
    return (1- SSIM_LOSS_RATIO)*l2_loss(output, target) + SSIM_LOSS_RATIO*compute_ssim_loss(output, target)





def get_infilled_image(img, encoder, decoder):
        fill_area = predict_fill_area_for_img(img, encoder, decoder)
        with_center = AddImageCenter(IMAGE_SIZE, CENTER_SIZE)(torch.clone(img), fill_area)
        return with_center





def show_sample_photos(encoder, decoder, title="Sample Batch Photos",):
    images = []
    for sample_image_index in range(0,3):
        cutout_image = sample_cutout[sample_image_index]
        predicted_image = get_infilled_image(cutout_image, encoder, decoder)
        images += [predicted_image, sample_gt[sample_image_index]]
    display_images(images, images_per_row=2, title=title)





def get_val_loss(encoder, decoder, loss_fn):
    with torch.no_grad():
        total_loss = 0
        number_of_images = len(val_loader)
        for batch in val_loader:
                images, gt_images, _ = batch
                images = images.to(device)
                gt_images = gt_images.to(device)

                embeded_images = encoder(images)
                decoded_images = decoder(embeded_images)
                total_loss += loss_fn(get_center_of_image(gt_images), decoded_images).item()
    return total_loss / number_of_images





def display_loss_history_graph(train_loss_history, val_loss_history, title="Training and Validation Loss" ):
    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)

    plt.figure()
    plt.plot(train_loss_history[:, 0], train_loss_history[:, 1], label="Training Loss")
    plt.plot(val_loss_history[:, 0], val_loss_history[:, 1], label="Validation Loss")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()






def do_training(epochs, train_loss_history, val_loss_history, data_loader, optim, encoder, decoder, loss_fn, val_interval = 10, should_save = False):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, verbose=True, T_max=10)
    # scheduler2 = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10, verbose=True)
    best_val = 999999
    for epoch in range(epochs):
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        encoder.train()
        decoder.train()
        
        epoch_loss = 0
        total_number_of_images = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=total_number_of_images)
        for i, batch in pbar:
            images, gt_images, _ = batch

            images = images.to(device)
            gt_images = gt_images.to(device)

            optim.zero_grad()

            embedded_images = encoder(images)
            decoded_images = decoder(embedded_images)

            loss = loss_fn(get_center_of_image(gt_images), decoded_images)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        epoch_loss /= total_number_of_images
        train_loss_history.append([epoch, epoch_loss])
        if epoch % val_interval == 0:
            val_loss = get_val_loss(encoder, decoder, loss_fn)
            print(f"Val Loss: {val_loss}")
            val_loss_history.append([epoch, val_loss])
            if val_loss < best_val and should_save:
                save_model(encoder, decoder, epoch, optim, f"./models/l2-model-{epoch}-{val_loss:.2f}.pth")
                best_val = val_loss
            # scheduler2.step(val_loss)

        if epoch % 5 == 0:
            clear_output()
            display_loss_history_graph(train_loss_history, val_loss_history, title="L2 Loss Training and Validation Loss")
            plt.savefig("./l2-model.png")


        # scheduler.step()
        print(f"Epoch Loss: {epoch_loss}")


# ## Do Sanity Check with single image dataset 


class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        self.sobel_filter_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_filter_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).broadcast_to((1, 3, 3, 3))
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).broadcast_to((1, 3, 3, 3))
        self.sobel_filter_x.weight = nn.Parameter(sobel_kernel_x)
        self.sobel_filter_y.weight = nn.Parameter(sobel_kernel_y)

    def forward(self, image):
        grad_output_x = self.sobel_filter_x(image)
        grad_output_y = self.sobel_filter_y(image)
        return grad_output_x, grad_output_y
sobel_filter = SobelFilter()
sobel_filter = sobel_filter.to(device)





sample_batch_img, sample_batch_img_gt, sample_batch_classes = next(iter(val_loader))
sample_train_batch_img, sample_train_batch_img_gt, sample_traing_batch_classes = next(iter(train_loader))

# Compute a loss based on the difference in edges of the images
def edge_diff_loss(img1, img2):
    grad_x_img_1, grad_y_img_1 = sobel_filter(img1)
    grad_x_img_2, grad_y_img_2 = sobel_filter(img2)

    grad_diff_x = torch.abs(grad_x_img_1 - grad_x_img_2)
    grad_diff_y = torch.abs(grad_y_img_1 - grad_y_img_2)

    loss = torch.mean(grad_diff_x + grad_diff_y)
    return loss





def l1_and_sharpness_loss(output, target):
    L1_LAMBDA = 15
    SHARPNESS_LAMBDA = 15
    # SSIM_LAMBDA = 0.8

    # L1_LAMBDA = 0
    # SHARPNESS_LAMBDA = 1
    SSIM_LAMBDA = 0

    # l1_loss_value = 0
    l1_loss_value = l2_loss(output, target)
    edge_diff_loss_value = edge_diff_loss(output, target)
    # edge_diff_loss_value = 0
    ssim_loss_value = 0
    # ssim_loss_value = ssim_loss(output, target)

    # print("L1 Loss:", l1_loss_value)
    # print("Edge Diff Loss:", edge_diff_loss_value)
    # print("SSIM Loss:", ssim_loss_value)
    
    # print("L1 Loss:", L1_LAMBDA * l1_loss_value)
    # print("Edge Diff Loss:", SHARPNESS_LAMBDA * edge_diff_loss_value)
    # print("SSIM Loss:", ssim_loss_value * SSIM_LAMBDA)

    return L1_LAMBDA*l1_loss_value+ SHARPNESS_LAMBDA*edge_diff_loss_value+ SSIM_LAMBDA*ssim_loss_value

sharpness_enc = Encoder()
sharpness_dec = Decoder()

optim = get_optimizer(sharpness_enc, sharpness_dec)

sharpness_train_loss_history = []
sharpness_val_loss_history = []
do_training(
  epochs=10000, 
  train_loss_history=sharpness_train_loss_history, 
  val_loss_history=sharpness_val_loss_history, 
  data_loader=train_loader, 
  optim=optim,
  encoder=sharpness_enc, 
  decoder=sharpness_dec,
  should_save=True,
  loss_fn=l1_and_sharpness_loss
)

