# Original training settings are preserved in this standalone historical file.
_base_ = ["icod_train_fder.py"]
model_name = "EffB1_ZoomNeXt_FDER"
pretrained = True
use_checkpoint = False
output_dir = "outputs_fder_restored_20260906"
info = "fder_original_upload_restored"
