#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 

#### ViT testing for DIF 

######### Install torch GPU version, pytorch-lightning and transformers ######### 
'''
Run this code segments only once to install the main packages such as torch and transformer
'''
# ! pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# ! pip install transformers pytorch-lightning --quiet

### Check torch version and GPU count ###
import torch
print(torch.__version__)
print(torch.cuda.device_count())

import requests
import math
import matplotlib.pyplot as plt
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError,ImageChops, ImageEnhance
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path
import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, HfFolder, Repository, notebook_login
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os
from torchmetrics.classification import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import itertools
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#pip install -U albumentations
import albumentations as A
#from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

# Read dataset path  
data_dir = Path('path to file\\data')

#ref: https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-10-albumentations-dataloader/notebook
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []
        self.root = root_dir
        
        if total_classes:
            self.classnames  = os.listdir(root_dir)[:total_classes] # for test
        else:
            self.classnames = os.listdir(root_dir)
            
        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)
            
            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, target = self.data[index]
        img = np.array(Image.open(data))
        
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]
        
        target = torch.from_numpy(np.array(target))
        img = torch.from_numpy(img)
        
        #print(type(img),img.shape, target)
        
        return img,target 

original_input = ImageFolder(data_dir)
print("Test data count: " +str(len(original_input)))

image, label = original_input[0]
plt.imshow(image)
print(label)

import skimage
from skimage import color

#Ref: https://www.kaggle.com/code/hirotaka0122/bengali-custom-albumentations-transforms
compression_quality = 90
tp = A.ImageCompression(quality_lower=compression_quality, quality_upper=compression_quality, always_apply=True, p=1)
 
def jpeg_diff(img):   
    img = np.uint8(skimage.color.rgb2ycbcr(img))
    compr_img = tp(image=img)['image']
    diff = ImageChops.difference(Image.fromarray(img.astype(np.uint8)), Image.fromarray(compr_img.astype(np.uint8))) 
    #print(type(diff))
    diff = np.asarray(diff)
    diff[:,:,0] = np.zeros((np.shape(diff[:,:,0])))
    add = ImageChops.add(Image.fromarray(img.astype(np.uint8)),  Image.fromarray(diff.astype(np.uint8)))
    
    extrema = add.getextrema()
    max_diff_add = max([ex[1] for ex in extrema])
    if max_diff_add == 0:
        max_diff_add = 1
    scale = 255.0 / max_diff_add  
    add = ImageEnhance.Brightness(add).enhance(scale)
    return np.asarray(add)

class DIFF(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False
        
    def apply(self, img, **params):
        return jpeg_diff(img)

a_transform = A.Compose(
    [
        DIFF()        
    ]
)


processed_input = ImageFolder(data_dir,total_classes=3, transform=a_transform)
print("Test data count: " +str(len(processed_input)))

image, label = processed_input[0]
plt.imshow(image)
print(label)


import os 
for i in range (len(prcessed_input)):
    filename = 'path to file\\test_folder\\' + str(i) + '.jpg'
    cv2.imwrite(filename, np.asarray(processed_input[i][0]))

type(prcessed_input[0])
test_data = processed_input

### Preparing Dataset Labels
label2id = {}
id2label = {}

for i, class_name in enumerate(test_data.classnames):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

print("class and label details: "+str(id2label))

### Image Classification Collator
'''
To apply our transforms to images, we'll use a custom collator class. 
We'll initialize it using an instance of ViTFeatureExtractor and pass the collator 
instance to torch.utils.data.DataLoader's collate_fn kwarg.
'''
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
 
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings 

# Init Feature Extractor, Model, Data Loaders
# For other models: https://huggingface.co/google/vit-large-patch16-224-in21k
# 1. google/vit-base-patch16-224-in21k
# 2. google/vit-large-patch16-224-in21k
pre_trained_model = 'google/vit-large-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(pre_trained_model)

model = ViTForImageClassification.from_pretrained(
    pre_trained_model,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

collator = ImageClassificationCollator(feature_extractor)

test_loader = DataLoader(test_data, batch_size=1, collate_fn=collator) #, num_workers=4

### Training Model Class 
class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss 

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss 

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"Loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"Acc", acc, prog_bar=True)
        return outputs.loss
    
    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        pred = outputs.logits.argmax(1)
        #print("prediction:"+ str(pred)+ "baatch ID: " + str(batch_idx))
        return pred
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

### Test from best checkpoint
pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
chkpt = 'path to file\\checkpoint_file_name.ckpt'
trainer = pl.Trainer(accelerator="gpu", devices=1, )
predictions = trainer.predict(classifier,test_loader, ckpt_path = chkpt)

### Converting prediction tensor into LIST of predictions
predicted = []
data = predictions[:]
for i in range(len(data)):
  temp = data[i][0].tolist()
  predicted.append(temp)

print("Length of prediction LIST: " +str(len(predicted)))

### Creating ground-truth labels form test loader 
batches = iter(test_loader)
tmp_labels = []
for i in range(len(batches)):
  tmp_data = next(batches) 
  key, data = tmp_data.items()
  tmp_labels.append(data[1].tolist())

gt_labels = list(itertools.chain(*tmp_labels))
print("Length of ground-truth LIST: " +str(len(gt_labels)))

### plot confusion matrix of the results 

class_names=['gan', 'graphic', 'real']
cm = confusion_matrix(y_target=gt_labels, 
                      y_predicted=predicted, 
                      binary=False)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_normed=True,
                                cmap="YlGnBu",
                                colorbar=True)#,
                                #class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')

### Test classification report 
test_samples = len(gt_labels) #2400
print('\nTest Classification Report\n')
test_rpt = classification_report(gt_labels, predicted, target_names=class_names)
print(test_rpt)

### Geting Accuracy for test Split 
trainer.test(classifier,test_loader)

