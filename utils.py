import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2, ToTensor

def concatenate_imgs(images: list[ImageFile], size: int=224) -> Image:
    """
    images : list of the 4th images of 1 house
    size : length of 1 side of resized image
    
    """

    resized_images = [image.resize((size, size)) for image in images]

    new_image = Image.new("RGB", (size*2, size*2))

    new_image.paste(resized_images[0], (0, 0))  # Top-left
    new_image.paste(resized_images[1], (size, 0))  # Top-right
    new_image.paste(resized_images[2], (0, size))  # Bottom-left
    new_image.paste(resized_images[3], (size, size))  # Bottom-right

    return new_image

def add_gaussian_noise(image: Image, mean: float=0, std: float=0.05):
    """
        Add a gaussian noise to disturb colors of the image.
    """
    
    np_image = np.array(image)  # Convert PIL image to numpy array
    noise = np.random.normal(mean, std, np_image.shape)  # Add Gaussian noise
    noisy_image = np.clip(np_image + noise * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def random_resized_crop(image: Image, min_scale: float=0.9, max_scale: float=1.1):
    """
        Resize the image with scale +- 10% of the original size. Then crop borders.
    """
    width, height = image.size
    scale_factor = np.random.uniform(min_scale, max_scale)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.BILINEAR)

def augment_imgs(images: list[ImageFile]) -> tuple[list[ImageFile]]:
    """
    From a list of 4 images, this function create a list of transformed images.
    Originals and Transformed images are then mixed to create two list of differents images.
    
    """
    
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=1),  # Horizontal flip with 50% chance
        v2.RandomRotation(degrees=10),   # Random rotation within ±10 degrees
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        v2.Lambda(lambda x: random_resized_crop(x, min_scale=0.9, max_scale=1.1)),  # Custom resizing function
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
        v2.Lambda(lambda x: add_gaussian_noise(x))  # Custom function to add noise
    ])

    transformed_imgs = [transform(img) for img in images]

    ## Mix original and transformed images

    mask = np.random.choice([0, 1], size=4)

    house1 = []
    house2 = []

    for p in range(len(images)):
        if mask[p] == 1:
            house1.append(images[p])
            house2.append(transformed_imgs[p])
        else:
            house1.append(transformed_imgs[p])
            house2.append(images[p])

    return house1, house2

def augment_all_images(list_images: list[list[ImageFile]]) -> list[list[ImageFile]]:
    """
    Return the list of all augmented images

    ### example :

    input : [
        [img1_1, img1_2, img1_3, img1_4],
        [img2_1, img2_2, img2_3, img2_4]
    ]

    output: [
        [augm_img1_1,  augm_img1_2,  img1_3,       img1_4     ], # house 1
        [img2_1,       img2_2,       img2_3,       augm_img2_4], # house 2
        [img1_1,       img1_2,       augm_img1_3,  augm_img1_4], # house 1
        [augm_img2_1,  augm_img2_2,  augm_img2_3,  img2_4     ], # house 2
    ]

    """
    
    house1 = []
    house2 = []

    for images in tqdm(list_images):
        h1, h2 = augment_imgs(images)
        house1.append(h1)
        house2.append(h2)

    list_augmented_images = house1 + house2

    return list_augmented_images
  
def extract_features(images: list[ImageFile], idxs: list[int], model, processor) -> pd.DataFrame:
    features_list = {}
    for idx, img in tqdm(zip(idxs, images), total=len(images)):
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)

        features_list[idx] = outputs.pooler_output.detach().numpy()[0].tolist()

    nb_features = len(features_list[next(iter(features_list))])
    features_df = pd.DataFrame.from_dict(features_list, orient='index', columns=[f"feature_{i+1}" for i in range(nb_features)])

    return features_df

class ImageStructuredDataset(Dataset):
    def __init__(self, image_list, structured_data, labels):
        """
        Args:
            image_list (list of PIL.Image): List of PIL images.
            structured_data (pd.DataFrame): DataFrame with structured features.
            labels (list or pd.Series): List or Series with labels.
        """
        self.transformers = ToTensor()
        self.image_list = [self.transformers(img) for img in image_list]
        self.structured_data = torch.tensor(structured_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).flatten()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img = self.image_list[idx]
        struct = self.structured_data[idx]
        label = self.labels[idx]

        return img, struct, label

class FromScratchModel(nn.Module):
    def __init__(self, img_output_dim, structured_input_dim, fc_hidden_dim, num_outputs):
        super(FromScratchModel, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to get a fixed size
        )
        
        # Fully connected layer for CNN output
        self.fc_img = nn.Linear(64*4*4, img_output_dim)
        
        # Fully connected layers for structured data
        self.fc_struct = nn.Sequential(
            nn.Linear(structured_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU()
        )
        
        # Final regression layer
        self.fc_combined = nn.Sequential(
            nn.Linear(img_output_dim + fc_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
    def forward(self, img, structured_data):
        # Image processing
        img_features = self.cnn(img)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = F.relu(self.fc_img(img_features))
        
        # Structured data processing
        struct_features = self.fc_struct(structured_data)
        
        # Combine both features
        combined_features = torch.cat((img_features, struct_features), dim=1)
        
        # Final regression output
        output = self.fc_combined(combined_features)

        return output
