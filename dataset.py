import pandas as pd
import re

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
from hanzidentifier import is_simplified


class HanziDataset(Dataset):

    def __init__(self, data_dir, data_file_name, wrap_size=32, font='STSONG'):
        '''data_dir (Path): path to the data directory
           transform: torchvision.transforms
        '''
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.Resize((wrap_size, wrap_size)),
             transforms.ToTensor()])
        self.data_df = pd.read_csv(data_dir / data_file_name)
        self.img_dir = data_dir / 'hanzi_img' / font

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.data_df.loc[idx, 'image_name']
        img = Image.open(str(img_path)).convert('L')
        img_tensor = self.transform(img)
        return img_tensor


def get_valid_chinese_chars(tokenizer, simplify=True):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    vocab = tokenizer.get_vocab()
    if simplify:
        chinese_chars = [
            char for char in vocab
            if pattern.match(char) and is_simplified(char)
        ]
    else:
        chinese_chars = [
            char for char in vocab
            if pattern.match(char) and not is_simplified(char)
        ]

    return chinese_chars