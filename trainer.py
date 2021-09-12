#built-in

import os
import numpy as np
# torch
import torch
from torch import optim
from torch.utils.data import DataLoader

# yolov4 package
from models import Yolov4
from dataset import Yolo_dataset
from losses import Yolo_loss


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


class Trainer(object):
    def __init__(self, config):
        self.config = config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create model
        self.model = Yolov4(n_classes=config.classes)
        self.model.cuda()

        # create optimizer
        if config.TRAIN_OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=config.learning_rate / config.batch,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    )
        elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                    lr=config.learning_rate / config.batch,
                                    momentum=config.momentum,
                                    weight_decay=config.decay,
                                    )
        
        # create scheduler
        def burnin_schedule(i):
            """learning rate setup
            """
            if i < config.burn_in:
                factor = pow(i / config.burn_in, 4)
            elif i < config.steps[0]:
                factor = 1.0
            elif i < config.steps[1]:
                factor = 0.1
            else:
                factor = 0.01
            return factor
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, burnin_schedule)

        self.criterion = Yolo_loss(device=device, batch=self.config.batch // self.config.subdivisions, n_classes=self.config.classes)

        self.init_pretrained()
        self.init_training_dataset()

    def init_training_dataset(self):
        """initialize the training dataset
        """
        self.train_dataset = Yolo_dataset(self.config.train_label, self.config, train=True)
        self.train_dataloader = DataLoader(self.train_dataset, 
                                            batch_size=self.config.batch // self.config.subdivisions, 
                                            shuffle=True,
                                            num_workers=8, 
                                            pin_memory=True, 
                                            drop_last=True,
                                            collate_fn=collate)

    def init_pretrained(self):
        """load the checkpoint, resume training
        """
        # read checkpoint
        if self.config.pretrained:
            ckp = torch.load(self.config.pretrained)
            # only use conv137
            if not ckp.get('model_state_dict'):
                model_weight = ckp
                self.model.init_conv137(model_weight)
            # read all tools
            else:
                self.model.load_state_dict(ckp['model_state_dict'])
                self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckp['scheduler_state_dict'])

    def save_ckp(self, epoch:int):
        """save the checkpoint
        """
        save_prefix = 'Yolov4_epoch'
        save_path = os.path.join(self.config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
        data = {'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}
        torch.save(data, save_path)
        

    def auto_delete(self, epoch:int):
        save_prefix = 'Yolov4_epoch'
        for i in range(1, epoch):
            if i % 5 == 0:
                continue
            save_path = os.path.join(self.config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
            if os.path.isfile(save_path):
                os.remove(save_path)

