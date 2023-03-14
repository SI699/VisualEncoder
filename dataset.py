import pandas as pd

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


class HanziDataset(Dataset):

    def __init__(self, data_dir, data_file_name, wrap_size=32):
        '''data_dir (Path): path to the data directory
           transform: torchvision.transforms
        '''
        self.data_dir = data_dir
        self.transform = transforms.compose(
            [transforms.Resize((wrap_size, wrap_size)),
             transforms.ToTensor()])
        self.data_df = pd.read_csv(data_dir / data_file_name)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_dir / 'hanzi_img' / self.data_df.iloc[idx,
                                                                   'image_name']
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        return img_tensor
