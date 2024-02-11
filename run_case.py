from mls.mls import MVC
import torch
import json
from utils.inference_tools import features_from_sensors
import sys
import argparse
import numpy as np


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Machine Vibration Classification")
    parser.add_argument("--model_f", type=str, required=True,
            help="Path to the model checkpoint file")
    parser.add_argument("--sample", type=str,
            help=".json file containing the input raw samples from sensors 'Sensor 1', 'Sensor 2' and 'Sensor 2' and class 'class'")
    parser.add_argument("--feature_norm", type=str,
            help=".json file containing normalization values of each feature")
    parser.add_argument("--freq_norm", type=str,
            help=".json file containing normalization values of each fft signal")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # model = MVC(312).double()
    # checkpoint = torch.load(args.model_f)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    sample = json.load(open(args.sample))
    signals = [np.array(sample['s0']), np.array(sample['s1']), np.array(sample['s2'])]
    features = features_from_sensors(signals, args.feature_norm, args.freq_norm)
    # pred = model.predict(features)
    # print(f"Classe: {sample['class']}")
    # print(f"Classe predita: {pred}")
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
