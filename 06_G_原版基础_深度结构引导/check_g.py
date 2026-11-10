"""Synthetic functional tests, not depth quality or COD accuracy evaluation."""
import copy,gc,json,tempfile
from pathlib import Path
import numpy as np
import torch
from mmengine import Config
from methods.zoomnext.depth_guided import DepthGuidedRGPU,EffB1_ZoomNeXt_DepthGuided
from methods.zoomnext.zoomnext import EffB1_ZoomNeXt
from methods.zoomnext.layers import RGPU

def active(m):return any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
def main():
    torch.set_num_threads(2);torch.manual_seed(16)
    original=RGPU(16).eval()
    block=DepthGuidedRGPU.from_original(RGPU(16)).eval()
    state=block.state_dict();state.update(original.state_dict());block.load_state_dict(state,strict=True)
    x=torch.randn(2,16,7,11)
    torch.testing.assert_close(block(x),original(x),rtol=0,atol=0)
    opt=torch.optim.SGD(block.parameters(),lr=.1)
    for step in range(2):
        opt.zero_grad();y,d=block(x,return_depth=True)
        ((y*torch.randn_like(y)).mean()+d.square().mean()).backward()
        assert active(block.depth_condition.delta)
        if step==1:assert active(block.depth_condition.q) and active(block.depth_condition.kv)
        opt.step()
    assert not torch.equal(block(x,guide=True),block(x,guide=False))
    print('PASS G: original RGPU initialization, depth-dependent gate effect, attention/gate gradients',flush=True)
    data={f'image_{k}':torch.rand(2,3,h,w) for k,h,w in [('l',96,144),('m',64,96),('s',32,48)]}
    data['mask']=(torch.rand(2,1,64,96)>.5).float();data['depth']=torch.rand(2,1,64,96)
    ref=EffB1_ZoomNeXt(pretrained=False).eval()
    off=EffB1_ZoomNeXt_DepthGuided(pretrained=False,use_depth=False).eval()
    off.load_state_dict(ref.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(off(data),ref(data),rtol=0,atol=0)
    ref.train();off.train()
    torch.manual_seed(99);a=ref(data,iter_percentage=.4)['loss']
    torch.manual_seed(99);b=off(data,iter_percentage=.4)['loss']
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    del ref,off,a,b;gc.collect()
    reports={}
    for name in ('baseline','g_aux','g_only'):
        cfg=Config.fromfile(f'configs/g_{name}.py')
        model=EffB1_ZoomNeXt_DepthGuided(pretrained=False,**cfg.model_cfg)
        if name!='baseline':
            try:model({k:v for k,v in data.items() if k!='depth'})
            except ValueError:pass
            else:raise AssertionError('Missing depth accepted')
        assert not any(any(t in n for t in ('context_proj','context_gates','context_scale','relation_gates','relation_scale')) for n in model.state_dict())
        calls={};handles=[]
        for stage in (5,4,3,2,1):
            def hook(m,i,o,stage=stage):calls[stage]=calls.get(stage,0)+1
            handles.append(getattr(model,f'hmu_{stage}').register_forward_hook(hook))
        res=model(data,iter_percentage=.4);assert torch.isfinite(res['loss']);res['loss'].backward()
        assert calls=={s:1 for s in (5,4,3,2,1)}
        for h in handles:h.remove()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        if name!='baseline':
            assert all(active(getattr(model,f'hmu_{s}').depth_condition.stem) and active(getattr(model,f'hmu_{s}').depth_condition.head) for s in range(1,6))
        model.eval()
        with torch.no_grad():
            clean=model({k:v for k,v in data.items() if k.startswith('image_')})
            changed=model(dict(data,depth=torch.zeros_like(data['depth']),mask=torch.zeros_like(data['mask'])))
        torch.testing.assert_close(clean,changed,rtol=0,atol=0)
        assert clean.shape==(2,1,64,96)
        reports[name]=dict(parameters=sum(p.numel() for p in model.parameters()),passed=True)
        print('PASS',name,'5 stages, loss/backward, RGB-only inference',flush=True)
        del model,res,grads;gc.collect()
    model=EffB1_ZoomNeXt_DepthGuided(pretrained=False).eval()
    with torch.no_grad():out=model({f'image_{k}':torch.rand(1,3,n,n) for k,n in [('l',576),('m',384),('s',192)]})
    assert out.shape==(1,1,384,384) and torch.isfinite(out).all()
    print('PASS RGB-only 384x384 forward',flush=True)
    # Exercise the actual augmented training dataset and strict pairing.
    import cv2
    import albumentations as A
    from main_for_image import ImageTrainDataset
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for name in ('Imgs','GT','Depth'):(root/name).mkdir()
        mask=np.zeros((32,48),np.uint8);mask[4:20,3:17]=255
        rgb=np.repeat(mask[:,:,None],3,2)
        cv2.imwrite(str(root/'Imgs/a.png'),rgb);cv2.imwrite(str(root/'GT/a.png'),mask)
        np.save(root/'Depth/a.npy',mask.astype(np.float32)/255)
        info=dict(root=td,image=dict(path='Imgs',suffix='.png'),mask=dict(path='GT',suffix='.png'),depth=dict(path='Depth',suffix='.npy'))
        ds=ImageTrainDataset({'toy':info},dict(h=32,w=48),require_depth=True)
        ds.trains=A.Compose([A.HorizontalFlip(p=1),A.RandomBrightnessContrast(p=1)])
        sample=ds[0]['data'];torch.testing.assert_close(sample['depth'],sample['mask'],rtol=0,atol=0)
        assert sample['image_m'].shape==(3,32,48)
        (root/'Depth/a.npy').unlink()
        try:ImageTrainDataset({'toy':info},dict(h=32,w=48),require_depth=True)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing paired depth was ignored')
    print('PASS actual loader: aligned flip, depth unaffected by color transforms, missing-file rejection',flush=True)
    Path('validation_g.json').write_text(json.dumps(dict(torch=torch.__version__,device='cpu',variants=reports,full_384_forward=True,real_teacher_tested=False,performance_verified=False),indent=2),encoding='utf-8')
if __name__=='__main__':main()
