import util
import json
import datetime
import os
from torch.optim import Adam

class SqueezeNetConfig(object):
    """Neural network training config to hold hyperparemters"""
    def __init__(self):
        self.DEVICE = 'cpu' #util.check_gpu()
        self.SEED = 10
        self.LR = 0.0005
        self.BATCH = 4
        self.EPOCHS = 100
        self.IMG_SIZE = (640, 360) # W x H
        
        self.WEIGHT_DECAY = 1e-4
        self.L1_LAMBDA = 0.001 
        
        self.ANCHOR_RATIO = [0.5, 1, 2]
        self.ANCHOR_SCALE = [64, 256, 2048]
        self.ANCHOR_BASE_SIZE = 16
        self.RPN_OUT = 256
        
        self.CONFIDENCE_THRESHOLD = 0.515
        self.NMS_THRESHOLD = 0.3
        
        with open('secrets.json', 'r') as file:
            secrets = json.load(file)
        file.close()
        
        self.TIMESTAMP = f'{datetime.datetime.now().strftime("%H%M%S_%f")}'

        self.BASE_PATH_TORCH = secrets['models']['SqueezeNetTorch']
        self.BASE_PATH_LOGS = secrets['models']['SqueezeNetLogs']

        if not os.path.exists(self.BASE_PATH_TORCH):
            os.makedirs(self.BASE_PATH_TORCH)
        if not os.path.exists(self.BASE_PATH_LOGS):
            os.makedirs(self.BASE_PATH_LOGS)

        self.CHECKPOINT = os.path.join(self.BASE_PATH_TORCH, f'{self.TIMESTAMP}.checkpoint')
        self.LOG = os.path.join(self.BASE_PATH_LOGS, f'{self.TIMESTAMP}')

        self.TRAIN = True
        self.TEST = True
            