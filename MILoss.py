import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as skm

def thirdOrderSplineKernel(u):
    abs_u = u.abs()
    sqr_u = abs_u.pow(2.0)

    result = torch.FloatTensor(u.size()).zero_()
    result = result.to(u.device)

    mask1 = abs_u<1.0
    mask2 = (abs_u>=1.0 )&(abs_u<2.0)

    result[mask1] = (4.0 - 6.0 * sqr_u[mask1] + 3.0 * sqr_u[mask1] * abs_u[mask1]) / 6.0
    result[mask2] = (8.0 - 12.0 * abs_u[mask2] + 6.0 * sqr_u[mask2] - sqr_u[mask2] * abs_u[mask2]) / 6.0

    return result

class MILoss(nn.Module):
    def __init__(self, num_bins=16):
        super(MILoss, self).__init__()
        self.num_bins = num_bins


    def forward(self, moving, fixed):
        moving = moving.view(moving.size(0), -1)
        fixed = fixed.view(fixed.size(0), -1)

        padding = float(2)
        batchsize = moving.size(0)

        fixedMin = fixed.min(1)[0].view(batchsize,-1)
        fixedMax = fixed.max(1)[0].view(batchsize,-1)

        movingMin = moving.min(1)[0].view(batchsize,-1)
        movingMax = moving.max(1)[0].view(batchsize,-1)
        #print(fixedMax,movingMax)

        JointPDF = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()
        movingPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        fixedPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        JointPDFSum = torch.FloatTensor(batchsize).zero_()
        JointPDF_norm = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()


        if JointPDF.device != moving.device:
            JointPDF = JointPDF.to(moving.device)
            movingPDF = movingPDF.to(moving.device)
            fixedPDF = fixedPDF.to(moving.device)
            JointPDFSum = JointPDFSum.to(moving.device)
            JointPDF_norm = JointPDF_norm.to(moving.device)

        #print(JointPDF.device)

        fixedBinSize = (fixedMax - fixedMin) / float((self.num_bins - 2 * padding))
        movingBinSize = (movingMax - movingMin) / float(self.num_bins - 2 * padding)

        fixedNormalizeMin = fixedMin / fixedBinSize - float(padding)
        movingNormalizeMin = movingMin / movingBinSize - float(padding)

        #print(fixed.shape,fixedBinSize.shape,fixedNormalizeMin.shape)
        fixed_winTerm = fixed / fixedBinSize - fixedNormalizeMin

        fixed_winIndex = fixed_winTerm.int()
        fixed_winIndex[fixed_winIndex < 2] = 2
        fixed_winIndex[fixed_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        moving_winTerm = moving / movingBinSize - movingNormalizeMin

        moving_winIndex = moving_winTerm.int()
        moving_winIndex[moving_winIndex < 2] = 2
        moving_winIndex[moving_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        #torch.autograd.set_detect_anomaly(True)

        for b in range(batchsize):
            a_1_index = moving_winIndex[b] - 1
            a_2_index = moving_winIndex[b]
            a_3_index = moving_winIndex[b] + 1
            a_4_index = moving_winIndex[b] + 2

            a_1 = thirdOrderSplineKernel((a_1_index - moving_winTerm[b]))
            a_2 = thirdOrderSplineKernel((a_2_index - moving_winTerm[b]))
            a_3 = thirdOrderSplineKernel((a_3_index - moving_winTerm[b]))
            a_4 = thirdOrderSplineKernel((a_4_index - moving_winTerm[b]))
            for i in range(self.num_bins):
                fixed_mask = (fixed_winIndex[b] == i)
                fixedPDF[b][i] = fixed_mask.sum()
                for j in range(self.num_bins):
                    JointPDF[b][i][j] = a_1[fixed_mask & (a_1_index == j)].sum() + a_2[
                        fixed_mask & (a_2_index == j)].sum() + a_3[fixed_mask & (a_3_index == j)].sum() + a_4[
                                            fixed_mask & (a_4_index == j)].sum()

            #JointPDFSum[b] = JointPDF[b].sum()
            #norm_facor = 1.0 / JointPDFSum[b]
            #print(JointPDF[b])
            #inverse_sum[b] = 1.0 / JointPDF[b].sum()
            #xx = JointPDF[b] / JointPDFSum[b]
            #print(xx)
            #print(JointPDF[b])
            JointPDF_norm[b] = JointPDF[b] / JointPDF[b].sum()
            fixedPDF[b] = fixedPDF[b] / fixed.size(1)

        movingPDF = JointPDF_norm.sum(1)
        
        #print(JointPDF_norm)

        MI_loss = torch.FloatTensor(batchsize).zero_().to(moving.device)
        for b in range(batchsize):
            JointPDF_mask = JointPDF_norm[b] > 0
            movingPDF_mask = movingPDF[b] > 0
            fixedPDF_mask = fixedPDF[b] > 0

            MI_loss[b] = (JointPDF_norm[b][JointPDF_mask] * JointPDF_norm[b][JointPDF_mask].log()).sum() \
                         - (movingPDF[b][movingPDF_mask] * movingPDF[b][movingPDF_mask].log()).sum() \
                         - (fixedPDF[b][fixedPDF_mask] * fixedPDF[b][fixedPDF_mask].log()).sum()

        #print(MI_loss)
        loss = MI_loss.mean()
        return -1.0*loss

def cc1(x1, x2):
    # Reshape tensors to [n*c, h*w]
    x1_flat = x1.view(x1.size(0) * x1.size(1), -1)
    x2_flat = x2.view(x2.size(0) * x2.size(1), -1)

    # Calculate mean-centered tensors
    x1_mean = x1_flat - torch.mean(x1_flat, dim=1, keepdim=True)
    x2_mean = x2_flat - torch.mean(x2_flat, dim=1, keepdim=True)

    # Calculate Pearson correlation coefficient
    numerator = torch.sum(x1_mean * x2_mean, dim=1)
    denominator = torch.sqrt(torch.sum(x1_mean ** 2, dim=1) * torch.sum(x2_mean ** 2, dim=1))
    correlation_coefficient = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero

    # Calculate correlation loss as negative mean of correlation coefficients
    loss = -torch.mean(correlation_coefficient)

    return loss

def cc(img1, img2):
    img1 = (img1 / torch.max(img1) ).to(torch.float32)
    img2 = (img2 / torch.max(img2)).to(torch.float32)
    # print('img1',torch.max(img1))
    # print('img2',torch.max(img2))
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    print('img1', torch.min(img1), torch.max(img1))
    print('img2', torch.min(img2), torch.max(img2))
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))

    cc = torch.clamp(cc, -1., 1.)
    print(cc.mean())
    return cc.mean()


def MI(cls, image_F, image_A, image_B):
    cls.input_check(image_F, image_A, image_B)
    return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),image_B.flatten())

class MI(nn.Module):
    def __init__(self):
        super(MI, self).__init__()

    @classmethod
    def input_check(cls, *images):
        for image in images:
            if not isinstance(image, torch.Tensor):
                raise ValueError("所有输入图像都必须是 torch.Tensor 对象。")
            if image.dim() != 4:
                raise ValueError("图像维度必须为 4，即 (N, C, H, W)。")

    @classmethod
    def forward(cls, image_F, image_A, image_B, num_bins=256, epsilon=1e-10):
        assert image_F.shape == image_A.shape == image_B.shape, "All images must have the same shape"

        num_samples, num_channels, height, width = image_F.shape

        # Flatten images and convert to numpy arrays
        image_F_np = image_F.view(num_samples, -1).detach().cpu().numpy()
        image_A_np = image_A.view(num_samples, -1).detach().cpu().numpy()
        image_B_np = image_B.view(num_samples, -1).detach().cpu().numpy()

        # Calculate histograms
        hist_F, _ = np.histogram(image_F_np, bins=num_bins, range=(0, 1))
        hist_A, _ = np.histogram(image_A_np, bins=num_bins, range=(0, 1))
        hist_B, _ = np.histogram(image_B_np, bins=num_bins, range=(0, 1))

        # Calculate joint histograms
        joint_hist_AF, _, _ = np.histogram2d(image_F_np.flatten(), image_A_np.flatten(), bins=num_bins, range=[[0, 1], [0, 1]])
        joint_hist_BF, _, _ = np.histogram2d(image_F_np.flatten(), image_B_np.flatten(), bins=num_bins, range=[[0, 1], [0, 1]])

        # Normalize histograms
        p_F = hist_F / np.sum(hist_F)
        p_A = hist_A / np.sum(hist_A)
        p_B = hist_B / np.sum(hist_B)
        p_AF = joint_hist_AF / np.sum(joint_hist_AF)
        p_BF = joint_hist_BF / np.sum(joint_hist_BF)

        # Calculate mutual information
        mi_A = np.sum(p_AF * np.log((p_AF + epsilon) / ((p_A[:, np.newaxis] + epsilon) * (p_F[np.newaxis, :] + epsilon))))
        mi_B = np.sum(p_BF * np.log((p_BF + epsilon) / ((p_B[:, np.newaxis] + epsilon) * (p_F[np.newaxis, :] + epsilon))))

        total_mi = mi_A + mi_B
        return -1.0 * total_mi