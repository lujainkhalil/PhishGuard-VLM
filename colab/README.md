# Google Colab

Open **`PhishGuard_VLM_Colab.ipynb`** in Colab (GPU runtime). It clones this repo under `/content/Phishguard-VLM`, installs CUDA PyTorch then `colab/requirements-no-torch.txt`, uses `configs/colab_training.yaml` (batch size 1, AMP), and saves checkpoints to Drive.

Training config overrides live in **`configs/colab_training.yaml`**; CLI flags in `scripts/run_train.py` include `--use-amp`, `--batch-size`, `--epochs`, `--eval-steps`, `--save-steps`.
