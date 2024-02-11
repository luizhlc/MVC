from utils import dataProcessing as dataP
import torch
# def feature_extractor(signals):
#     n=200
#     w=n
#     f=10000
#     for s_data in signals:
#         s_data = dataP.fill_the_gaps(s_data):
#         grms = dataP.get_RMS(s_data, f, n, w)
#         peak2peak = get_peak2peak(s_data)
#         peak = get_peak(s_data)
#         get_crista(peak, rms_global)
#         _, ffts_y = apply_fft(s_data, n)

# def feature_normalization(features, feat_norm_f, fft_norm_f):
#     ft_limits = json.load(open(feat_norm_f))
#     for m in features:
#         for s, s_data in enumerate(features[m]):
#             min_v, max_v = ft_limits[m][f's{s}']


def features_from_sensors(signals):
    return torch.rand(1,312).double()