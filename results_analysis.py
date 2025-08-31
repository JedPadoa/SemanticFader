import os
import re
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from regCAV import RegCAV
import torch
from transformers import ClapModel, ClapProcessor


def compute_pair_results(original_file: str,
                         autoencoded_file: str,
                         RCVs) -> Dict[Tuple[str, str], list]:
	# Parse bias from autoencoded filename: "bias_{attr}_{value}"
	filename = os.path.basename(autoencoded_file)
	m = re.search(r"bias_([a-zA-Z]+)_([\-+]?\d*\.?\d+)", filename)
	bias_attr = m.group(1) if m else None
	bias_val = float(m.group(2)) if m else 0.0

	# Unpack RegCAV objects
	speed_regcav, grass_regcav = RCVs

	# Compute attribute scores with RegCAV (autoencoded − original)
	speed_orig = speed_regcav.compute_attribute_score(original_file)
	speed_ae = speed_regcav.compute_attribute_score(autoencoded_file)
	grass_orig = grass_regcav.compute_attribute_score(original_file)
	grass_ae = grass_regcav.compute_attribute_score(autoencoded_file)

	return {
		(original_file, autoencoded_file): [
			bias_val,
			{
				"speed_original": speed_orig,
				"speed_autoencoded": speed_ae,
				"grass_original": grass_orig,
				"grass_autoencoded": grass_ae,
				"grass_delta": float(grass_ae - grass_orig),
				"speed_delta": float(speed_ae - speed_orig) 
			}
		]
	}


def analyze_output_dir_and_plot(output_dir: str,
                                RCVs) -> Dict[Tuple[str, str], list]:
	# Find original (unbiased) file
	wavs = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(".wav")]
	orig = None
	# Prefer explicit 'output_unbiased.wav'
	preferred = os.path.join(output_dir, "output_unbiased.wav")
	if os.path.exists(preferred):
		orig = preferred
	if orig is None:
		# Fallback: any file with 'unbiased' and without 'bias_' in name
		candidates = [p for p in wavs if "unbiased" in os.path.basename(p).lower() and "bias_" not in os.path.basename(p).lower()]
		if candidates:
			orig = sorted(candidates)[0]
	if orig is None:
		# Last resort: any file without 'bias_' in name
		candidates = [p for p in wavs if "bias_" not in os.path.basename(p).lower()]
		if candidates:
			orig = sorted(candidates)[0]
	if orig is None:
		raise FileNotFoundError("No unbiased/original file found in directory.")

	# Iterate biased outputs
	results: Dict[Tuple[str, str], list] = {}
	for p in wavs:
		name = os.path.basename(p)
		if p == orig:
			continue
		if "bias_" not in name:
			continue
		# Only process files matching "bias_{attr}_{value}"
		if not re.search(r"bias_([a-zA-Z]+)_([\-+]?\d*\.?\d+)", name):
			continue
		res = compute_pair_results(orig, p, RCVs)
		results.update(res)

	# Build plotting lists: each file on x-axis, label as "original" or "added {attribute}"
	labels: List[str] = []
	speed_vals: List[float] = []
	grass_vals: List[float] = []

	# Add original once (take from any result payload)
	if results:
		any_payload = next(iter(results.values()))
		_, attrs = any_payload
		labels.append("original")
		speed_vals.append(attrs["speed_original"])
		grass_vals.append(attrs["grass_original"])

		# Add each biased file
	# Sort for stable display
	for (o, ae), payload in sorted(results.items(), key=lambda kv: os.path.basename(kv[0][1])):
		bias_amount, attrs = payload
		ae_name = os.path.basename(ae)
		m = re.search(r"bias_([a-zA-Z]+)_([\-+]?\d*\.?\d+)", ae_name)
		bias_attr = (m.group(1).lower() if m else "unknown")
		labels.append(f"added {bias_attr} (bias + {bias_amount})")
		speed_vals.append(attrs["speed_autoencoded"])
		grass_vals.append(attrs["grass_autoencoded"])

	# Plot: files on x-axis, attribute scores on y-axis
	x = np.arange(len(labels))

	# Speed figure
	plt.figure(figsize=(max(8, len(labels) * 0.6), 4))
	plt.scatter(x, speed_vals, alpha=0.8, color="black", marker="x")
	plt.title("Speed attribute score per file")
	plt.ylabel("predicted speed")
	plt.ylim(0.0, 1.5)
	plt.xticks(x, labels, rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig("results_speed.png")
	plt.close()

	# Grass figure
	plt.figure(figsize=(max(8, len(labels) * 0.6), 4))
	plt.scatter(x, grass_vals, alpha=0.8, color="black", marker="x")
	plt.title("Grass attribute score per file")
	plt.ylabel("predicted grass")
	plt.ylim(0.0, 1.5)
	plt.xticks(x, labels, rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig("results_grass.png")
	plt.close()


if __name__ == "__main__":
	# Example usage: share one CLAP across both RegCAVs
	device = "cuda" if torch.cuda.is_available() else "cpu"
	clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device).eval()
	clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

	speed_regcav = RegCAV(clap_model=clap_model, clap_processor=clap_processor, device=device)
	grass_regcav = RegCAV(clap_model=clap_model, clap_processor=clap_processor, device=device)
	speed_regcav.load_rcv("RCVs/speed_rcv.pkl")
	grass_regcav.load_rcv("RCVs/grass_rcv.pkl")

	output_dir = "/Users/jeddo/Documents/sf_rave_sg_v3_80k"
	analyze_output_dir_and_plot(output_dir, (speed_regcav, grass_regcav))