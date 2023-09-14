from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis
import pyiqa

def loss_function(vis_image, ir_image ,output_image, device):
    
    def loss_cal1(input, output, device):
        SSIM_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        MS_SSIM_func = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        PSNR_func = PeakSignalNoiseRatio(data_range=1.0).to(device)

        ssim_loss = (1 - SSIM_func(output, input))*100
        ms_ssim_loss = (1- MS_SSIM_func(output, input))*100
        psnr_loss = -PSNR_func(output, input)

        return ssim_loss, ms_ssim_loss, psnr_loss

    # def loss_cal2(output, device):

    #     # loss_func = pyiqa.create_metric('clipiqa', device=device, as_loss=True)
    #     # hyperiqa_func = pyiqa.create_metric('hyperiqa', device=device, as_loss=True)
    #     loss_func = pyiqa.create_metric('brisque', device=device, as_loss=True)

    #     loss_nr = loss_func(output)/100
    #     # hyperiqa_loss = hyperiqa_func(output)

    #     return loss_nr

    ssim_loss_vis, ms_ssim_loss_vis, psnr_loss_vis = loss_cal1(vis_image, output_image, device)
    ssim_loss_ir, ms_ssim_loss_ir, psnr_loss_ir = loss_cal1(ir_image, output_image, device)

    ssim_loss_avg = (ssim_loss_vis + ssim_loss_ir)/2
    ms_ssim_loss_avg = (ms_ssim_loss_vis + ms_ssim_loss_ir)/2
    psnr_loss_avg = (psnr_loss_vis + psnr_loss_ir)/2

    loss_nr = loss_cal2(output_image, device)

    loss = (ssim_loss_avg + ms_ssim_loss_avg + psnr_loss_avg + loss_nr)/4

    # print('Loss:',loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, clipiqa_loss)

    return loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, loss_nr