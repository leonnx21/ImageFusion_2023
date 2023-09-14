# import pyiqa
# import torch

# # list all available metrics
# print(pyiqa.list_models())

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # # # create metric with default setting
# # brisque = pyiqa.create_metric('brisque', device=device)
# # # # Note that gradient propagation is disabled by default. set as_loss=True to enable it as a loss function.
# # iqa_loss = pyiqa.create_metric('ssim', device=device, as_loss=True)

# # # create metric with custom setting
# # iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)

# # # check if lower better or higher better
# # print('Lower Is',iqa_loss.lower_better)

# # # example for iqa score inference
# # # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# # score_fr = iqa_metric(img_tensor_x, img_tensor_y)
# # score_nr = iqa_metric(img_tensor_x)

# # # img path as inputs.
# # score_fr = iqa_metric('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')

# # # For FID metric, use directory or precomputed statistics as inputs
# # # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
# # fid_metric = pyiqa.create_metric('fid')
# # score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
# # score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")

# SSIM_func = pyiqa.create_metric('ssim', device=device, as_loss=True)
# MS_SSIM_func = pyiqa.create_metric('ms_ssim', device=device, as_loss=True)
# PSNR_func = pyiqa.create_metric('psnr', device=device, as_loss=True)
# brisque_func = pyiqa.create_metric('brisque', device=device, as_loss=True)

# print('SSIM Lower Is better: ', SSIM_func.lower_better)
# print('MS_SSIM Lower Is better: ', MS_SSIM_func.lower_better)
# print('PSNR_func Lower Is better: ', PSNR_func.lower_better)
# print('brisque_func Lower Is better: ', brisque_func.lower_better)

# import necessary library
import torch

# create tensors
T1 = torch.Tensor([[1,2],[3,4]])
T2 = torch.Tensor([[0,3],[4,1]])
T3 = torch.Tensor([[4,3],[2,5]])

# print above created tensors
print("T1:\n", T1)
print("T2:\n", T2)
print("T3:\n", T3)

print(T1.shape)

# print("join(concatenate) tensors in the 0 dimension")
# T = torch.cat((T1,T2,T3), 0)
# print("T:\n", T)

print("join(concatenate) tensors in the -1 dimension")
T = torch.cat((T1,T2,T3), 2)
print("T:\n", T)