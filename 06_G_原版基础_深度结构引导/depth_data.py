"""Synchronized RGB/mask/pseudo-depth augmentation; depth is TRAIN-only."""
from pathlib import Path
import numpy as np
import cv2
import torch
from utils import io,ops

def load_depth(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Missing pseudo-depth: {path}')
    d=np.load(path,allow_pickle=False) if path.suffix=='.npy' else cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if d is None or d.ndim!=2 or not np.isfinite(d).all():raise ValueError(f'Invalid depth: {path}')
    if np.issubdtype(d.dtype,np.integer):d=d.astype(np.float32)/np.iinfo(d.dtype).max
    d=d.astype(np.float32)
    if d.min()<0 or d.max()>1:raise ValueError(f'Depth must be normalized to [0,1]: {path}')
    return d

def make_depth_dataset(base):
    class DepthTrainDataset(base):
        def __init__(self,dataset_infos,shape,require_depth=True):
            super().__init__(dataset_infos,shape)
            self.require_depth=require_depth;self.depth_paths={}
            if not require_depth:return
            for item in dataset_infos.values():
                if 'depth' not in item:raise ValueError('Each training dataset needs depth: {path: ..., suffix: .npy}')
                root=Path(item['root']);folder=root/item['depth']['path']
                images=(root/item['image']['path']).resolve()
                for image,_ in self.total_data_paths:
                    p=Path(image)
                    if p.parent.resolve()==images:
                        dp=folder/(p.stem+item['depth']['suffix'])
                        if not dp.is_file():raise FileNotFoundError(dp)
                        self.depth_paths[image]=dp
            if len(self.depth_paths)!=len(self.total_data_paths):raise ValueError('Depth pairing incomplete or duplicated RGB paths')

        def __getitem__(self,index):
            if not self.require_depth:return super().__getitem__(index)
            image_path,mask_path=self.total_data_paths[index]
            image=io.read_color_array(image_path);mask=io.read_gray_array(mask_path,thr=0)
            depth=load_depth(self.depth_paths[image_path])
            if depth.shape!=image.shape[:2]:raise ValueError(f'Depth must match original RGB resolution: {image_path}')
            if image.shape[:2]!=mask.shape:
                h,w=mask.shape;image=ops.resize(image,height=h,width=w)
                depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_LINEAR)
            # Same geometric transform; photometric transforms apply ONLY to RGB.
            # Both mask-like targets use nearest-neighbor in Rotate, as upstream.
            sample=self.trains(image=image,masks=[mask,depth])
            image=sample['image'];mask,depth=sample['masks']
            h,w=self.shape['h'],self.shape['w']
            images=ops.ms_resize(image,scales=(.5,1.,1.5),base_h=h,base_w=w)
            data={f'image_{k}':torch.from_numpy(np.ascontiguousarray(v)).div(255).permute(2,0,1) for k,v in zip(('s','m','l'),images)}
            data['mask']=torch.from_numpy(ops.resize(mask,height=h,width=w)).unsqueeze(0)
            data['depth']=torch.from_numpy(cv2.resize(depth,(w,h),interpolation=cv2.INTER_LINEAR).clip(0,1)).unsqueeze(0)
            return dict(data=data)
    return DepthTrainDataset
