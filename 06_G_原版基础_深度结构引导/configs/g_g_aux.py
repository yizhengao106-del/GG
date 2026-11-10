_base_ = ['icod_train_fder_restored.py']
model_name='EffB1_ZoomNeXt_DepthGuided'
model_cfg={'use_depth': True, 'use_depth_guidance': False, 'depth_loss_weight': 0.1}
load_from=None
output_dir='outputs_g_original'
info='original_g_aux'
