from utils import dataProcessing as dataP
import torch
import numpy as np
import json

# Não me orgulho pelo fato desse código ser uma cópia redundante dos outros scripts com uma adaptação técnica para reutilizar as funções
# Mas neste ponto, meu cérebro já não está mais funcionando muito bem depois de tantas horas de trabalho sem pausa

def features_from_sensors(sensor_data, feature_file, freq_file):
    features = torch.zeros(1,312).double()
    n = 200
    freq = 10000
    rms_global = []
    peak = []
    peak2peak =[]
    crista = []
    fft_ys = []
    for i in range(3):
        sensor_data[i] = dataP.fill_the_gaps(sensor_data[i])
        
    for s, s_data in enumerate(sensor_data):
        rms = dataP.get_RMS(np.asarray([s_data]), freq, n, n)
        rms_global.append(rms.flatten())
        peak.append(dataP.get_peak(np.asarray([s_data])))
        crista.append(dataP.get_crista(peak[s], rms))
        peak2peak.append(dataP.get_peak2peak(np.asarray([s_data])))
        _, yfs = dataP.apply_fft(np.asarray([s_data]), freq, n)
        fft_ys.append(yfs)
    metrics_map = {
        "RMS": rms_global, 
        "Peak": peak, 
        "Peak2Peak": peak2peak, 
        "Crista": crista
    }
    metrics_norm_map = {
        "RMS": [], 
        "Peak": [], 
        "Peak2Peak": [], 
        "Crista": []
    }
    
    print(rms_global, peak, crista, peak2peak)
    
    ft_limits = json.load(open(feature_file))
    for m in metrics_map:
        results = []
        for s, s_data in enumerate(metrics_map[m]):
            min_v, max_v = ft_limits[m][f's{s}']
            result = (metrics_map[m][s]-min_v)/(max_v-min_v)
            metrics_norm_map[m].append(result)
    fft_ys_norm = []
    freq_limits = json.load(open(freq_file))
    for s, yfs in enumerate(fft_ys):
        max_v = freq_limits[f's{s}']
        fft_ys_norm.append(yfs/max_v)
    features = []
    features.append(
        metrics_norm_map["RMS"][0][0]
    )
    for s in range(n_sensors):
        for m, m_data in metrics_norm_map.items():
    for label, f_data in metrics_norm_map.items():
        for s_idx, s_data in enumerate(f_data):
            features.append(metrics_norm_map[label][s_idx]
    s0_RMS
    s0_Peak
    s0_Peak2Peak
    s0_Crista
    s1_RMS
    s1_Peak
    s1_Peak2Peak
    s1_Crista
    s2_RMS
    s2_Peak
    s2_Peak2Peak
    s2_Crista
    print(fft_ys_norm)
    print(metrics_norm_map)

































            
    return features