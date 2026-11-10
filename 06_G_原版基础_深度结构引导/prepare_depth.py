"""Offline pseudo-depth export. Teacher is optional DA-V2, NOT OVCoser's claimed teacher."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-cfg',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='depth-anything/Depth-Anything-V2-Small-hf')
    p.add_argument('--revision',default='main');p.add_argument('--device',default='cuda:0')
    p.add_argument('--datasets',nargs='+',default=['cod10k_tr','camo_tr'])
    args=p.parse_args()
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    source=Path(args.data_cfg).resolve();cfg=yaml.safe_load(source.read_text(encoding='utf-8'))
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Use a new output folder to avoid mixing pseudo-depth versions')
    # Resolve relative roots against the provided dataset.yaml, independent of cwd.
    for item in cfg.values():
        if isinstance(item,dict) and 'root' in item:
            root=Path(item['root']).expanduser()
            item['root']=str(root.resolve() if root.is_absolute() else (source.parent/root).resolve())
    processor=AutoImageProcessor.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForDepthEstimation.from_pretrained(args.model,revision=args.revision).to(args.device).eval()
    out.mkdir(parents=True)
    metadata=dict(model=args.model,revision=args.revision,resolved_commit=getattr(model.config,'_commit_hash',None),normalization='per-image min-max [0,1], float32 npy',images={})
    for name in args.datasets:
        item=cfg[name];root=Path(item['root']);folder=out/name;folder.mkdir()
        images=root/item['image']['path'];masks=root/item['mask']['path']
        names=sorted({p.stem for p in images.glob('*'+item['image']['suffix'])}&{p.stem for p in masks.glob('*'+item['mask']['suffix'])})
        if not names:raise ValueError(f'No paired images for {name}')
        for stem in names:
            image_path=images/(stem+item['image']['suffix'])
            image=Image.open(image_path).convert('RGB')
            inputs=processor(images=image,return_tensors='pt').to(args.device)
            with torch.no_grad():
                pred=model(**inputs).predicted_depth.unsqueeze(1)
                pred=torch.nn.functional.interpolate(pred,size=image.size[::-1],mode='bicubic',align_corners=False)[0,0].float()
            if not torch.isfinite(pred).all() or pred.max()-pred.min()<1e-8:raise ValueError(f'Invalid/constant teacher output: {image_path}')
            arr=((pred-pred.min())/(pred.max()-pred.min())).cpu().numpy().astype(np.float32)
            np.save(folder/(stem+'.npy'),arr,allow_pickle=False)
            metadata['images'][name+'/'+stem]=hashlib.sha256(image_path.read_bytes()).hexdigest()
        item['depth']=dict(path=str(folder),suffix='.npy')
        print(name,len(names),'exported',flush=True)
    (out/'dataset_with_depth.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    (out/'teacher_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Use:',out/'dataset_with_depth.yaml')
if __name__=='__main__':main()
