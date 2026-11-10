"""OVCoser-inspired depth auxiliary learning, adapted to ORIGINAL RGPU output modulation.

Upstream: lartpang/OVCamo, commit 419c16beea40a9cf0458199f1172b8bdd37d505d.
Not a reproduction of the open-vocabulary model. See references/ and README.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .layers import RGPU
from .zoomnext import EffB1_ZoomNeXt
from .ops import resize_to
from .ovcamo_loss import l1_ssim_loss


class DepthStructureCondition(nn.Module):
    """RGB-derived depth features and channel cross-attention condition.

    Adapts OVCoser's depth stem + single-auxiliary StructureEnhancementBlock
    normalized channel Q/K attention. Uses GroupNorm instead of upstream LN2d;
    outputs a condition for group gates, not a direct image-feature residual.
    """
    def __init__(self, in_dim, hidden, groups):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_dim, hidden, 3, padding=1),
                                  nn.GroupNorm(1,hidden),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1))
        self.head = nn.Conv2d(hidden,1,1)
        self.rgb_proj = nn.Conv2d(in_dim,hidden,1)
        self.q_norm=nn.GroupNorm(1,hidden);self.kv_norm=nn.GroupNorm(1,hidden)
        self.q=nn.Conv2d(hidden,hidden,1,bias=False)
        self.kv=nn.Conv2d(hidden,hidden*2,1,bias=False)
        self.heads=4 if hidden%4==0 else 1
        self.out=nn.Conv2d(hidden,hidden,1)
        self.delta=nn.Conv2d(hidden,groups*hidden,1)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)

    def forward(self,x,predict_depth=False):
        depth=self.stem(x)
        b,c,h,w=depth.shape
        q=self.q(self.q_norm(self.rgb_proj(x))).reshape(b,self.heads,c//self.heads,h*w)
        k,v=self.kv(self.kv_norm(depth)).chunk(2,1)
        k=k.reshape_as(q);v=v.reshape_as(q)
        with torch.autocast(device_type=x.device.type,enabled=False):
            weights=(F.normalize(q.float(),dim=-1)@F.normalize(k.float(),dim=-1).transpose(-1,-2)).softmax(-1)
            mixed=(weights@v.float()).reshape(b,c,h,w)
        correction=self.delta(self.out(mixed.to(depth.dtype)))
        return correction,self.head(depth) if predict_depth else None


class DepthGuidedRGPU(RGPU):
    @classmethod
    def from_original(cls,original):
        block=cls.__new__(cls)
        nn.Module.__init__(block)
        block.num_groups=original.num_groups
        for name,module in original.named_children():block.add_module(name,module)
        hidden=original.interact['0'].conv.in_channels
        block.depth_condition=DepthStructureCondition(original.expand_conv.conv.in_channels,hidden,block.num_groups)
        return block

    def forward(self,x,guide=True,return_depth=False):
        delta,depth=self.depth_condition(x,predict_depth=return_depth)
        xs=self.expand_conv(x).chunk(self.num_groups,1)
        outs,gates=[],[]
        out,fork,gate=self.interact['0'](xs[0]).chunk(3,1);outs.append(out);gates.append(gate)
        for i in range(1,self.num_groups-1):
            out,fork,gate=self.interact[str(i)](torch.cat([xs[i],fork],1)).chunk(3,1)
            outs.append(out);gates.append(gate)
        out,gate=self.interact[str(self.num_groups-1)](torch.cat([xs[-1],fork],1)).chunk(2,1)
        outs.append(out);gates.append(gate)
        # Preserve the original sequential group interactions; only condition its
        # final channel-modulation logits. Softmax stays over all expanded channels.
        gate_logits=torch.cat(gates,1)
        for layer in list(self.gate_genator.children())[:-1]:gate_logits=layer(gate_logits)
        modulation=self.gate_genator[-1](gate_logits+delta if guide else gate_logits)
        out=self.final_relu(self.fuse(torch.cat(outs,1)*modulation)+x)
        return (out,depth) if return_depth else out


class EffB1_ZoomNeXt_DepthGuided(EffB1_ZoomNeXt):
    def __init__(self,pretrained=False,use_depth=True,use_depth_guidance=True,depth_loss_weight=.1,num_frames=1,**kwargs):
        if num_frames!=1:raise ValueError('Image-only model')
        if depth_loss_weight<0:raise ValueError('depth_loss_weight must be nonnegative')
        super().__init__(pretrained=pretrained,num_frames=num_frames,**kwargs)
        self.use_depth=bool(use_depth);self.use_depth_guidance=bool(use_depth_guidance)
        self.depth_loss_weight=float(depth_loss_weight)
        if self.use_depth:
            for stage in (5,4,3,2,1):
                name=f'hmu_{stage}'
                setattr(self,name,DepthGuidedRGPU.from_original(getattr(self,name)))

    def body_outputs(self,data):
        lf,mf,sf=[self.normalize_encoder(data[f'image_{k}']) for k in ('l','m','s')]
        x=None;depths=[]
        for stage in (5,4,3,2,1):
            tra=getattr(self,f'tra_{stage}')
            l,m,s=[tra(feats[stage-1]) for feats in (lf,mf,sf)]
            merged=getattr(self,f'siu_{stage}')(l=l,m=m,s=s)
            if x is not None:merged=merged+resize_to(x,tgt_hw=merged.shape[-2:])
            block=getattr(self,f'hmu_{stage}')
            if self.use_depth:
                result=block(merged,guide=self.use_depth_guidance,return_depth=self.training)
                if self.training:x,depth=result;depths.append(depth)
                else:x=result
            else:x=block(merged)
        return self.predictor(x),depths

    def body(self,data):return self.body_outputs(data)[0]

    def forward(self,data,iter_percentage=1,**kwargs):
        if self.training and self.use_depth:
            if 'depth' not in data:raise ValueError('G training requires RGB-aligned pseudo-depth data[depth]; no mask/gray fallback')
            d=data['depth']
            if d.shape!=data['mask'].shape or not torch.isfinite(d).all() or d.min()<0 or d.max()>1:
                raise ValueError('depth must match mask shape, be finite and normalized to [0,1]')
        logits,depths=self.body_outputs(data)
        if not self.training:return logits
        prob=logits.sigmoid()
        bce=F.binary_cross_entropy_with_logits(logits,data['mask'])
        coef=self.get_coef(iter_percentage=iter_percentage,method='cos',milestones=(0,1))
        ual=coef*(1-(2*prob-1).abs().pow(2)).mean()
        loss=bce+ual;vis=dict(sal=prob)
        text=f'bce: {bce.item():.5f} powual_{coef:.5f}: {ual.item():.5f}'
        if depths:
            with torch.autocast(device_type=logits.device.type,enabled=False):
                dl=torch.stack([l1_ssim_loss(v.float(),data['depth'].float()) for v in depths]).mean()
            loss=loss+self.depth_loss_weight*dl
            text+=f' depth_{self.depth_loss_weight:.3f}: {dl.item():.5f}'
            vis['depth']=resize_to(depths[-1],tgt_hw=logits.shape[-2:]).sigmoid()
        return dict(vis=vis,loss=loss,loss_str=text)
