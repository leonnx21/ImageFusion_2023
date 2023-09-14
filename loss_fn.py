from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis


def loss_function(vis_image, ir_image ,output_image, device):
    
    def loss_cal1(input, output, device):
        SSIM_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        MS_SSIM_func = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        PSNR_func = PeakSignalNoiseRatio(data_range=1.0).to(device)
        egras_func = ErrorRelativeGlobalDimensionlessSynthesis().to(device)

        ssim_loss = (1 - SSIM_func(output, input))*100
        ms_ssim_loss = (1- MS_SSIM_func(output, input))*100
        psnr_loss = -PSNR_func(output, input)
        egras_loss = egras_func(output, input)

        return ssim_loss, ms_ssim_loss, psnr_loss, egras_loss

    # def loss_cal2(output, device):

    #     tv_loss = qnr(output)

    #     return tv_loss


    ssim_loss_vis, ms_ssim_loss_vis, psnr_loss_vis, egras_loss_vis = loss_cal1(vis_image, output_image, device)
    ssim_loss_ir, ms_ssim_loss_ir, psnr_loss_ir, egras_loss_ir = loss_cal1(ir_image, output_image, device)

    # tv_loss = loss_cal2(output_image, device)

    ssim_loss_avg = (ssim_loss_vis + ssim_loss_ir)/2
    ms_ssim_loss_avg = (ms_ssim_loss_vis + ms_ssim_loss_ir)/2
    psnr_loss_avg = (psnr_loss_vis + psnr_loss_ir)/2
    egras_loss_avg = (egras_loss_vis + egras_loss_ir)/2

    loss = (ssim_loss_avg + ms_ssim_loss_avg + psnr_loss_avg)/3

    # print('LOSS {} {} {} {} {}'.format(loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg))

    return loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg