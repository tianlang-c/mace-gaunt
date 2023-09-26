import torch
import torch.fft as fft

def FFT_batch_channel(fs1, fs2, return_real = False):
    '''
    Calculate convolution g = fs1 * fs2
    Given two Fourier Series, represented by there fourier series coefficients,
    return the convolution result representd by its fourier series coefficients,
    i.e., g^*_{s,t}=\sum_{s_1+s_2=s,t_1+t_2=t}fs1^*_{1,s_1,t_1}fs2^*_{2,s_2,t_2}.
    Input: fs1, fs2 are 2D tensor that represent Fourier Series 
    FS_i = \sum_{s_i,t_i}fsi[s_i,t_i]e^(jcos(s_i\theta + t_i\phi)), i={1,2}
    Output: g is 2D tensor that represent Fourier Series 
    G = \sum_{s,t}g[s,t]exp(jcos(s\theta + t\phi))
    '''
    
    # Step 0: preparation
    B, C = fs1.shape[0], fs1.shape[1]
    in_shape1, in_shape2 = fs1.shape[2], fs2.shape[2]
    out_shape = in_shape1 + in_shape2 -1
    in1 = torch.zeros((B, C, out_shape, out_shape), dtype = fs1.dtype, device = fs1.device)
    in2 = torch.zeros((B, C, out_shape, out_shape), dtype = fs2.dtype, device = fs2.device)
    in1[:, :, :in_shape1, :in_shape1] = fs1
    in2[:, :, :in_shape2, :in_shape2] = fs2
    
    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fs1_freq, fs2_freq = fft.fft2(in1), fft.fft2(in2)
    
    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fs1_freq * fs2_freq
    
    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)
    
    return res


def sh2fs_batch_channel(sh_coeff, Y):
    '''
    Convert Spherical Harmonics coefficients to Fourier Series coefficients. (Batch + Channel version)
    Input: 
        sh_coeff: Spherical Harmonics coefficients, shape (B, C, L, 2L-1)
        Y: Precomputed Fourier Series coefficients for Spherical Harmonics Basis, shape (L, 2L-1, 2L-1, 2)
    Output:
        fs_coeff: Fourier Series coefficients, shape (B, C, 2L-1, 2L-1)
    '''
    
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * Y.unsqueeze(0).unsqueeze(0)).sum(dim=2) # (B, C, 2L-1, 2L-1, 2)
    fs_coeff = ((sum_along_L[:, :, :, :, 0] + sum_along_L[:, :, :, :, 1].flip(dims=[2]))).permute(0, 1, 3, 2) # (B, C, 2L-1, 2L-1)
    return fs_coeff


def fs2sh_batch_channel(fs_coeff, H):
    '''
    Convert Fourier Series coefficients to Spherical Harmonics coefficients. (Batch + Channel version)
    Input: 
        fs_coeff: Fourier Series coefficients, shape (B, C, 2L-1, 2L-1)
        H: Precomputed Spherical Harmonics coefficients for Fourier Series Basis, shape (L, 2L-1, 2L-1, 2)
    Output:
        sh_coeff: Spherical Harmonics coefficients, shape (B, C, L, 2L-1)
    '''
    
    fs_coeff_t_first = fs_coeff.permute(0, 1, 3, 2)
    sum_positive = (fs_coeff_t_first.unsqueeze(2) * H[:, :, :, 0].unsqueeze(0).unsqueeze(0)).sum(dim=-1) # (B, C, L, 2L-1)
    sum_negative = (fs_coeff_t_first.flip(dims=[2]).unsqueeze(2) * H[:, :, :, 1].unsqueeze(0).unsqueeze(0)).sum(dim=-1) # (B, C, L, 2L-1)
    sh_coeff = sum_positive + sum_negative
    return sh_coeff
