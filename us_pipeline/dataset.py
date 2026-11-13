"""Lazy memmap reads and on-demand rendering; no cached image tensor dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .images import render_window,prediction_basis


class ImageDataset:
    def __init__(self,prepared,split):
        if split not in ('train','validation','test'):
            raise ValueError('Unknown split')
        self.root=Path(prepared)
        self.metadata=json.loads((self.root/'manifest.json').read_text())
        if self.metadata['status']!='complete':
            raise ValueError('Incomplete prepared dataset')
        self.config=self.metadata['config']
        self.samples=pd.read_parquet(self.root/f'{split}.parquet')
        self._features=self._multipliers=None

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state=self.__dict__.copy()
        state['_features']=state['_multipliers']=None
        return state

    def window(self,index):
        if self._features is None:
            self._features=np.load(self.root/'features.npy',mmap_mode='r')
            self._multipliers=np.load(self.root/'multipliers.npy',mmap_mode='r')
        row=self.samples.iloc[index]
        s,t=int(row.security_index),int(row.end_index)
        return prediction_basis(self._features[s,t-self.config['lookback']+1:t+1],self._multipliers[s,t])

    def image(self,index):
        return render_window(self.window(index),self.config['include_volume'])

    def __getitem__(self,index):
        import torch
        image=self.image(index).astype('float32')[None,:,:]/255.
        return torch.from_numpy(image),int(self.samples.iloc[index].label),index
