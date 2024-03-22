import torch
# path='/data1/szy/model_zoo/ntire2024/ntirefinal/NTIRE2024_ImageSR_x4-main/model_zoo/sr_grl_base_c3x4.ckpt'
path='/data1/szy/model_zoo/ntire2024/ntirefinal/NTIRE2024_ImageSR_x4-main/model_zoo/DAT_x4.pth'
ckpt=torch.load(path,map_location="cpu")
# from models.team14_HGD import DAT
# model= DAT()
# model.load_state_dict(ckpt['params'],strict=True)
print(ckpt.keys())

