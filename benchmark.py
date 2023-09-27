from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, ErrorRelativeGlobalDimensionlessSynthesis 
import torch.nn as nn

class CustomBenchmark(nn.Module):
    def __init__(self):
            super(CustomBenchmark, self).__init__()

    def benchmark_function(self, vis_image, ir_image ,output_image, device):

        def benchmark_cal1(input, output, device):
            SSIM_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            MS_SSIM_func = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, normalize='relu').to(device)
            PSNR_func = PeakSignalNoiseRatio(data_range=1.0).to(device)
            egras_func = ErrorRelativeGlobalDimensionlessSynthesis().to(device)

            ssim = SSIM_func(output, input)
            ms_ssim = MS_SSIM_func(output, input)
            psnr = PSNR_func(output, input)
            egras_loss = egras_func(output, input)

            return ssim, ms_ssim, psnr, egras_loss

        ssim_vis, ms_ssim_vis, psnr_vis, egras_vis = benchmark_cal1(vis_image, output_image, device)
        ssim_ir, ms_ssim_ir, psnr_ir, egras_ir = benchmark_cal1(ir_image, output_image, device)

        ssim_avg = (ssim_vis + ssim_ir)/2
        ms_ssim_avg = (ms_ssim_vis + ms_ssim_ir)/2
        psnr_avg = (psnr_vis + psnr_ir)/2
        egras_avg = (egras_vis + egras_ir)/2

        # print('LOSS {} {} {} {} {}'.format(loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg))

        return ssim_avg, ms_ssim_avg, psnr_avg, egras_avg

    def forward(self, vis_image, ir_image , output_image, device):
        ssim_avg, ms_ssim_avg, psnr_avg, egras_avg = self.benchmark_function(vis_image, ir_image, output_image, device)
        return ssim_avg, ms_ssim_avg, psnr_avg, egras_avg



