# WMCodec-M2 (FIT5230 Milestone 2)

The models included in this repository are based on the WMCodec baseline model and have added functions such as logging, evaluation metrics, and attack simulation on the basis of the original inference code.

---

## 📂 Repository Structure

```text
WMCodec-M2/
└── WMCodec/                  # Source code
    ├── inference-at.py       # Enhanced inference script (Milestone 2 modifications)
    ├── requirements.txt
    ├── models.py
    ├── modules/
    ├── quantization/
    └── ...
```

## 🚀 How to Run in Colab

1. Clone code and model:
```bash
!git clone https://github.com/tianjingwen2000/WMCodec-M2.git
!git lfs install
!git clone https://huggingface.co/zjzser/WMCodec WMCodec_hc
```
2. Install dependencies:
```
%cd WMCodec-M2/WMCodec
!pip install -r requirements.txt
```
3. Upload the audio file for testing
```
uploaded = files.upload()
```
4. Run inference (example: Gaussian noise attack, SNR=20dB):
```
!python inference-at.py \
  --input_wavs_dir ../../input_audio \
  --output_dir ../../results_demo \
  --checkpoint_file ../../WMCodec_hc/save_model/g_00150000 \
  --run_id m2-gauss20 \
  --attack_type gaussian \
  --snr_db 20 \
  --seed 2025
```

## 🛠️ Milestone 2 Modifications

- Logging & Metrics: save results to results.csv (cross-entropy, bit accuracy, PESQ, STOI).

- Attack Simulation: added --attack_type gaussian and --snr_db to simulate Gaussian noise on input.

- Reproducibility: added --seed, fixed NumPy/Random/Torch seeds.

- Visualization: automatically generate plots: results_summary_clip_N.png (no-clip case); results_summary_clip_Y.png (clip case).

- Robustness: prevent divide-by-zero, clean NaN/Inf values before plotting.

- Sample Size: increased sample_num from 10 → 100 for stronger evaluation.

## 📊 Example Outputs

- results.csv: per-sample metrics

- results_summary_clip_N.png

- results_summary_clip_Y.png

## 📜Reference

Baseline implementation from [WMCodec](https://huggingface.co/zjzser/WMCodec).
