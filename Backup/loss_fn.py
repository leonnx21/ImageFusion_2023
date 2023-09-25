from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
            super(CustomLoss, self).__init__()

    def loss_function(self, vis_image, ir_image ,output_image, device):

        def loss_cal1(input, output, device):
            SSIM_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            MS_SSIM_func = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, normalize='relu').to(device)
            PSNR_func = PeakSignalNoiseRatio(data_range=1.0).to(device)
            # egras_func = ErrorRelativeGlobalDimensionlessSynthesis().to(device)

            ssim_loss = 1 - SSIM_func(output, input)
            ms_ssim_loss = 1 - MS_SSIM_func(output, input)
            psnr_loss = 1 - PSNR_func(output, input)/48
            # egras_loss = egras_func(output, input)
            # vif_loss = piq.VIFLoss(sigma_n_sq=2.0, data_range=256.)(input, output)

            return ssim_loss, ms_ssim_loss, psnr_loss

        # def loss_cal2(output, device):

        #     brisque_loss = piq.BRISQUELoss(data_range=256., reduction='none')(output)

        #     return brisque_loss

        ssim_loss_vis, ms_ssim_loss_vis, psnr_loss_vis = loss_cal1(vis_image, output_image, device)
        ssim_loss_ir, ms_ssim_loss_ir, psnr_loss_ir = loss_cal1(ir_image, output_image, device)

        ssim_loss_avg = (ssim_loss_vis + ssim_loss_ir)/2
        ms_ssim_loss_avg = (ms_ssim_loss_vis + ms_ssim_loss_ir)/2
        psnr_loss_avg = (psnr_loss_vis + psnr_loss_ir)/2

        loss = ssim_loss_avg + ms_ssim_loss_avg

        # print('LOSS {} {} {} {} {}'.format(loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg))

        return loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg


    def forward(self, vis_image, ir_image , output_image, device):
        loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg = self.loss_function(vis_image, ir_image, output_image, device)
        return loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg



