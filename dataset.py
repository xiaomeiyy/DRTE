import pickle
import json

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms



### input data of train and val, glove word embedding for labels
coco_train_file = 'dataset/coco_train.json'
coco_val_file = 'dataset/coco_val.json'
cocolabel_glove_file = 'dataset/cocolabels_glove.pkl'


### total categories
categories_sorted_by_freq = ['person', 'chair', 'car', 'dining table', 'cup',
                             'bottle', 'bowl', 'handbag', 'truck', 'backpack',
                             'bench', 'book', 'cell phone', 'sink', 'tv', 'couch',
                             'clock', 'knife', 'potted plant', 'dog', 'sports ball',
                             'traffic light', 'cat', 'bus', 'umbrella', 'tie', 'bed',
                             'fork', 'vase', 'skateboard', 'spoon', 'laptop',
                             'train', 'motorcycle', 'tennis racket', 'surfboard',
                             'toilet', 'bicycle', 'airplane', 'bird', 'skis', 'pizza',
                             'remote', 'boat', 'cake', 'horse', 'oven', 'baseball glove',
                             'baseball bat', 'giraffe', 'wine glass', 'refrigerator',
                             'sandwich', 'suitcase', 'kite', 'banana', 'elephant',
                             'frisbee', 'teddy bear', 'keyboard', 'cow', 'broccoli', 'zebra',
                             'mouse', 'orange', 'stop sign', 'fire hydrant', 'carrot',
                             'apple', 'snowboard', 'sheep', 'microwave', 'donut', 'hot dog',
                             'toothbrush', 'scissors', 'bear', 'parking meter', 'toaster',
                             'hair drier']


### category word embedding and label number
categories = categories_sorted_by_freq ## frequency first
cocolabel_glove = pickle.load(open(cocolabel_glove_file, 'rb'))
category_dict_classification = dict((category, count) for count, category in enumerate(categories))
labels_glove = [cocolabel_glove[key] for key in category_dict_classification]

def augmenter(image):
    return transforms.RandomHorizontalFlip(p=0.5)(
        transforms.ColorJitter(contrast=0.25)(
            transforms.RandomAffine(
                0, translate=(0.03, 0.03))(image)))


class COCOMultiLabel(Dataset):
    def __init__(self, args, train, image_path):
        super(COCOMultiLabel, self).__init__()
        self.train = train
        self.max_length = args.max_length
        if self.train == True:
            self.coco_json = json.load(open(coco_train_file, 'r'))   
            self.image_path = image_path + '/train2014/'
        else:
            self.coco_json = json.load(open(coco_val_file, 'r'))
            self.image_path = image_path + '/val2014/'
            
        self.fns = self.coco_json.keys()
        
    def __len__(self):
        return len(self.coco_json)
    
    
    def __getitem__(self, idx):
        ## get image 
        json_key = list(self.fns)[idx]       
        image_fn = self.image_path + json_key
        
        'load original image'
        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.train:
            try:
                image = augmenter(image)
            except IOError:
                print("augmentation error")
        transform=transforms.Compose([ transforms.Resize((448, 448)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        try:
            image = transform(image)        
        except IOError:
            return None
        
        ## get labels
        categories_batch = self.coco_json[json_key]['categories']
        labels_cls = np.zeros(len(categories), dtype=np.float32)
        labels_steps = np.zeros(self.max_length, dtype=np.int) + len(categories)
        label_length = 0
        for i in range(len(categories_batch)):
            category = categories_batch[i]
            labels_steps[i] = category_dict_classification[category]
            labels_cls[category_dict_classification[category]] = 1
            label_length += 1
            if len(categories_batch) > self.max_length:
                break
        labels_steps.sort()
        
        return (image, labels_cls, labels_steps, label_length, np.array(labels_glove, dtype=np.float32),json_key)
            