import pickle as pk
import numpy as np
import os
import json

results_dir = '/scratch/jtc440/ambisonic_source_separation/testing/results'
output_path = '/scratch/jtc440/ambisonic_source_separation/testing/results/snr.json'

bkgr_to_snr_list = {}

for fname in os.listdir(results_dir):
    fpath = os.path.join(results_dir, fname)

    with open(fpath, 'rb') as f:
        res = pk.load(f)
    snr = res['snr']



    bkgr_name = fname[9:].replace('_results.pkl', '')
    if bkgr_name not in bkgr_to_snr_list:
        bkgr_to_snr_list[bkgr_name] = []

    bkgr_to_snr_list[bkgr_name].append(snr)

total_snr_list = []
for bkgr_name, snr_list in bkgr_to_snr_list.items():
    mean_snr = np.mean(snr_list)
    total_snr_list += snr_list

    print("{}: {} dB".format(bkgr_name, mean_snr))

print()
print("Overall: {} dB".format(np.mean(total_snr_list)))

