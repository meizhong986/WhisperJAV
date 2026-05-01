"""
Cohere Transcribe spike — Step 1: verify model loadability.

Purpose: confirm transformers can import the Cohere ASR custom class and
download the model. No inference yet. If this fails, the spike stops here
and we drop Cohere from the v1.9.x ASR roster.

Pre-flight only — runs the absolute minimum to surface integration blockers:
  - Class import path
  - HF Hub authentication requirement
  - Custom class registration (trust_remote_code)
  - VRAM footprint estimate from model.dtype after load
"""
from __future__ import annotations

import sys
import time
import traceback


def main() -> int:
    print("=" * 70)
    print("Cohere Transcribe spike — Step 1: model loadability check")
    print("=" * 70)
    print()

    # ---- Environment baseline -------------------------------------------
    import torch
    import transformers

    print(f"transformers: {transformers.__version__}")
    print(f"torch:        {torch.__version__}")
    print(f"cuda:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:          {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram free:    {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB total")
    print()

    # ---- Try to access Cohere ASR class ---------------------------------
    print("Step 1a: import path probe")
    cls_native = None
    try:
        from transformers import CohereAsrForConditionalGeneration  # type: ignore[attr-defined]
        cls_native = CohereAsrForConditionalGeneration
        print(f"  OK  native: transformers.CohereAsrForConditionalGeneration -> {cls_native}")
    except (ImportError, AttributeError) as e:
        print(f"  --  not native to transformers: {e}")
        print("      -> will need trust_remote_code=True at load time")
    print()

    # ---- Probe model card metadata --------------------------------------
    print("Step 1b: HF Hub metadata probe (no full download yet)")
    model_id = "CohereLabs/cohere-transcribe-03-2026"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)
        print(f"  OK  model_id:     {info.id}")
        print(f"      sha:          {info.sha[:12] if info.sha else 'n/a'}")
        print(f"      gated:        {info.gated}")
        print(f"      private:      {info.private}")
        siblings = info.siblings or []
        weight_files = [s for s in siblings if s.rfilename.endswith((".bin", ".safetensors"))]
        total_weight_bytes = sum(s.size or 0 for s in weight_files)
        print(f"      weight files: {len(weight_files)} ({total_weight_bytes / 1024**3:.2f} GB total)")
        for s in weight_files[:5]:
            print(f"        - {s.rfilename}: {(s.size or 0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"  FAIL model info fetch failed: {e}")
        print("      Check network or HF auth (cohere may require gated access)")
        return 2
    print()

    # ---- Attempt model load --------------------------------------------
    print("Step 1c: model load (this is where the bytes get pulled)")
    print("  Note: first-time pull will download several GB. Be patient.")
    sys.stdout.flush()

    t0 = time.time()
    try:
        from transformers import AutoProcessor, AutoModel
        # Try generic AutoModel path first (works if Cohere ASR is upstream)
        if cls_native is not None:
            processor = AutoProcessor.from_pretrained(model_id)
            model = cls_native.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            load_method = "native"
        else:
            # Fall back to trust_remote_code path
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            load_method = "trust_remote_code"
        elapsed = time.time() - t0
        print(f"  OK  model loaded in {elapsed:.1f}s via {load_method} path")
        print(f"      model class: {type(model).__name__}")
        print(f"      parameters:  {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")
        print(f"      dtype:       {next(model.parameters()).dtype}")
        print(f"      device:      {next(model.parameters()).device}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"      vram used:   {allocated:.2f} GB after load")
    except Exception as e:
        print(f"  FAIL model load failed after {time.time() - t0:.1f}s: {e}")
        print()
        traceback.print_exc()
        return 3

    print()
    print("=" * 70)
    print("Step 1 SUCCESS — Cohere model loads in our environment.")
    print("Proceed to Step 2 (inference on 293s reference clip).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
