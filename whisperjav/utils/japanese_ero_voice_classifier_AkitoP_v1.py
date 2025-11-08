import argparse
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, HfApi, HfFolder
import os
import requests

import torch
from torch import nn

from whisperjav.utils.device_detector import get_best_device


# モデルの定義
class AudioClassifier(nn.Module):
    def __init__(
        self,
        label2id: dict,
        feature_dim=256,
        hidden_dim=256,
        device="cpu",
        dropout_rate=0.5,
        num_hidden_layers=2,
    ):
        super(AudioClassifier, self).__init__()
        self.num_classes = len(label2id)
        self.device = device
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        # 最初の線形層と活性化層を追加
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )
        # 隠れ層の追加
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout_rate),
            )
            self.hidden_layers.append(layer)
        # 最後の層（クラス分類用）
        self.fc_last = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        # 最初の層を通過
        x = self.fc1(x)

        # 隠れ層を順に通過
        for layer in self.hidden_layers:
            x = layer(x)

        # 最後の分類層
        x = self.fc_last(x)
        return x

    def infer_from_features(self, features):
        # 特徴量をテンソルに変換
        features = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # モデルを評価モードに設定
        self.eval()

        # モデルの出力を取得
        with torch.no_grad():
            output = self.forward(features)

        # ソフトマックス関数を適用して確率を計算
        probs = torch.softmax(output, dim=1)

        # ラベルごとの確率を計算して大きい順に並べ替えて返す
        probs, indices = torch.sort(probs, descending=True)
        probs = probs.cpu().numpy().squeeze()
        indices = indices.cpu().numpy().squeeze()
        return [(self.id2label[i], p) for i, p in zip(indices, probs)]

    def infer_from_file(self, file_path):
        feature = extract_features(file_path, device=self.device)
        return self.infer_from_features(feature)


from pyannote.audio import Inference, Model

emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(emb_model, window="whole")


def extract_features(file_path, device="cpu"):
    inference.to(torch.device(device))
    return inference(file_path)


def get_hf_token():
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print("[DEBUG] Found Hugging Face token in environment.")
        return hf_token
    print("Hugging Face token not found in environment variable HF_TOKEN.")
    print("If the repository is private, you need a token. You can get one at https://huggingface.co/settings/tokens")
    user_token = input("Enter your Hugging Face token (leave blank if not needed): ").strip()
    if user_token:
        print("[DEBUG] Using token from user input.")
        return user_token
    print("[DEBUG] No token provided; will attempt unauthenticated access.")
    return None

def validate_hf_token(token):
    if not token:
        return False
    try:
        api = HfApi()
        user = api.whoami(token=token)
        print(f"[DEBUG] Token validated for user: {user.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        return False

HF_REPO_ID = "AkitoP/Japanese-Ero-Voice-Classifier"
CKPT_FILENAME = "model_final.pth"
CONFIG_FILENAME = "config.json"

def get_model_and_config(cache_dir, device):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    HF_REPO_ID = "AkitoP/Japanese-Ero-Voice-Classifier"
    CKPT_FILENAME = "model_final.pth"
    CONFIG_FILENAME = "config.json"

    hf_token = get_hf_token()
    if hf_token and not validate_hf_token(hf_token):
        print("[ERROR] Provided Hugging Face token is invalid or does not have access.")
        print("Please check your token at https://huggingface.co/settings/tokens")
        exit(1)

    print(f"[DEBUG] Hugging Face repo: {HF_REPO_ID}")
    print(f"[DEBUG] Using token: {'SET' if hf_token else 'NOT SET'}")

    # Download config.json
    config_path = cache_dir / CONFIG_FILENAME
    if not config_path.exists():
        print(f"Downloading {CONFIG_FILENAME} from Hugging Face...")
        try:
            config_path = Path(
                hf_hub_download(repo_id=HF_REPO_ID, filename=CONFIG_FILENAME, cache_dir=str(cache_dir), token=hf_token)
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {CONFIG_FILENAME} from {HF_REPO_ID}.")
            print(f"[ERROR] Exception: {e}")
            print("[DIAGNOSTICS] Possible causes:")
            print("- The repository does not exist or is renamed/deleted.")
            print("- The repository is private and the token is missing or invalid.")
            print("- The file name is incorrect or missing in the repo.")
            print("[ACTION] Verify the repo URL in your browser. If private, check your token at https://huggingface.co/settings/tokens.")
            exit(1)
    config = json.loads(config_path.read_text())

    # Download model_final.pth
    ckpt_path = cache_dir / CKPT_FILENAME
    if not ckpt_path.exists():
        print(f"Downloading {CKPT_FILENAME} from Hugging Face...")
        try:
            ckpt_path = Path(
                hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_FILENAME, cache_dir=str(cache_dir), token=hf_token)
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {CKPT_FILENAME} from {HF_REPO_ID}.")
            print(f"[ERROR] Exception: {e}")
            print("[DIAGNOSTICS] Possible causes:")
            print("- The repository does not exist or is renamed/deleted.")
            print("- The repository is private and the token is missing or invalid.")
            print("- The file name is incorrect or missing in the repo.")
            print("[ACTION] Verify the repo URL in your browser. If private, check your token at https://huggingface.co/settings/tokens.")
            exit(1)

    print(f"Loading model from {ckpt_path}...")
    model = AudioClassifier(device=device, **config["model"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

def classify_audio(model, audio_file):
    print(f"Classifying {audio_file}...")
    output = model.infer_from_file(audio_file)
    print("Predicted:")
    for label, prob in output:
        print(f"{label}: {prob:.4f}")
    return output

def main():
    try:
        parser = argparse.ArgumentParser(description="Japanese Ero Voice Classifier CLI")
        parser.add_argument("audio_file", type=str, help="Path to the audio file to classify")
        parser.add_argument("--cache_dir", type=str, default="hf_cache", help="Hugging Face cache directory")
        args = parser.parse_args()

        # Use smart device detection: CUDA → MPS → CPU
        device = get_best_device()
        print(f"Device: {device}")

        model = get_model_and_config(args.cache_dir, device)
        classify_audio(model, args.audio_file)
    except Exception as e:
        print("[FATAL ERROR] An uncaught exception occurred.")
        print(f"[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
        print("[DIAGNOSTICS] Please check the above traceback and error messages for guidance.")
        print("If you are accessing a private or gated Hugging Face repo, ensure your token is correct and has access.")
        print("If the repo is public, verify the repo URL and file names.")
        exit(1)

if __name__ == "__main__":
    main()