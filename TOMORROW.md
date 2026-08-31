# Tomorrow — getting the best algae + bleaching detection (reef path)

**Goal:** on the bigger GPU, push the reef path's algae and bleaching accuracy
as far as it goes without a huge time sink. The reef path is one pass of a
pretrained model (`EPFL-ECEO/coralscapes-vit-b-dpt`) — you improve it by using a
bigger checkpoint, higher resolution, test-time augmentation, and (optionally)
fine-tuning. **No colony-detector training is involved here** — that's a separate
path for tank photos.

## Current baseline (small GPU, ViT-B, res 1536)

| signal | IoU | precision | recall |
|---|---|---|---|
| algae (class 10) | 0.560 | 0.618 | 0.857 |
| bleached coral (classes 4,16,19,33) | 0.704 | 0.749 | 0.922 |
| colony detector (aquarium path, `coral_soft`) | mAP@50 0.859 | | |

Measured by `python src/eval/evaluate_all.py` against Coralscapes val (166 images).

---

## Part A — get it running on the new machine (~20 min)

- [ ] **A1. Copy the project + dataset.**
  - the repo folder `NARSS_PROJECT-main/`
  - the Coralscapes dataset folder `D:\coralscapes\` (~5.5 GB) — the whole thing, keep the
    nested layout `.../coralscapes/coralscapes/{leftImg8bit,gtFine,classes.json}`

- [ ] **A2. Create a fresh venv** (don't copy `.venv/` between machines — it breaks):
  ```powershell
  cd <repo>
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  ```

- [ ] **A3. Install PyTorch for the new GPU's CUDA.** Check `nvidia-smi` (top-right shows
  CUDA version), then pick the matching wheel index:
  ```powershell
  # example for CUDA 12.8; use cu121 / cu124 / cu128 to match the machine
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```
  Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

- [ ] **A4. Hugging Face auth + gated access.**
  ```powershell
  huggingface-cli login          # paste a NEW read token from hf.co/settings/tokens
  ```
  Then in a browser, logged into the same HF account, click **"Agree and access"** on BOTH:
  - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m  (for ViT-B)
  - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m  (for ViT-L — do this now)

- [ ] **A5. Point the code at Coralscapes.** Set the env var to the folder that
  directly contains `leftImg8bit/` and `gtFine/`:
  ```powershell
  $env:CORALSCAPES_ROOT = "D:\coralscapes\coralscapes"
  $env:PYTHONPATH = "src"
  ```
  (or edit `_ROOT_CANDIDATES` in `src/data/coralscapes_to_yolo_seg.py` /
  `src/data/coralscapes_to_yolo_det.py` / `src/eval/evaluate_all.py`)

- [ ] **A6. Rebuild the eval masks + confirm the baseline reproduces:**
  ```powershell
  python src\data\coralscapes_to_yolo_seg.py     # writes data/processed/algae_seg/
  python src\eval\evaluate_all.py                 # ~10-15 min; expect ~0.56 / ~0.70 IoU
  ```
  If those numbers come back, the machine is set up correctly. Move on.

---

## Part B — accuracy upgrades, no training (~30 min of edits + eval time)

Do these one at a time and re-run the eval after each so you know what helped.

- [ ] **B1. Bigger checkpoint (ViT-L).** In `src/models/algae_segmenter.py`, change:
  ```python
  REPO_ID = "EPFL-ECEO/coralscapes-vit-l-dpt"    # was: coralscapes-vit-b-dpt
  ```
  First run re-downloads ~1.2 GB. If it errors with a gated-repo/403, you missed the
  `dinov3-vitl16` licence in A4 — go accept it, then retry.
  Expected: +2-4 IoU on both signals.

- [ ] **B2. Higher inference resolution.** Same file, change:
  ```python
  FULL_MAX_LONG_SIDE = 2048     # was 1536  (native Coralscapes width)
  ```
  Uses more VRAM, slower per image — fine on the big GPU. Helps thin/patchy algae most.

- [ ] **B3. Test-time augmentation (TTA).** In `src/models/algae_segmenter.py`, add this
  function next to `_run`, and make `segment_reef` call it instead of `_run`:
  ```python
  def _run_tta(model, im_rgb_uint8, max_long_side, scales=(0.75, 1.0, 1.25)):
      """Average softmax over a few scales + horizontal flip, then argmax."""
      arr = np.asarray(im_rgb_uint8)
      h0, w0 = arr.shape[:2]
      acc = None
      for s in scales:
          _, prob = _run(model, arr, int(max_long_side * s))          # (C, h, w)
          _, prob_f = _run(model, arr[:, ::-1], int(max_long_side * s))
          prob_f = prob_f[:, :, ::-1]
          for p in (prob, prob_f):
              pr = np.stack([
                  np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR))
                  for c in p
              ]).astype(np.float32) / 255.0
              acc = pr if acc is None else acc + pr
      pred_full = acc.argmax(0).astype(np.int16)
      return pred_full, acc
  ```
  Then in `segment_reef`, replace `pred, _ = _run(model, arr, FULL_MAX_LONG_SIDE)` with
  `pred, _ = _run_tta(model, arr, FULL_MAX_LONG_SIDE)`.
  Expected: +1-2 IoU. ~6x slower per image (fine for eval, maybe too slow for live demo —
  keep a flag).

- [ ] **B4. Re-run `python src\eval\evaluate_all.py`** and record the new
  `reef_segmenter` rows. Note: with ViT-L + res 2048 + TTA the reef eval will take
  ~45-60 min for all 166 images.

---

## Part C — what "good" looks like

| | ViT-B / 1536 (today) | realistic target (ViT-L / 2048 / TTA) | hard ceiling |
|---|---|---|---|
| algae IoU | 0.56 | 0.62 - 0.68 | ~0.70 (fuzzy category) |
| bleached IoU | 0.70 | 0.75 - 0.80 | ~0.85 |

If you're not near the target after B1-B3, the gap is **domain** — your photos differ
from Red Sea reef scenes — and only Part D closes it.

---

## Part D — fine-tune (optional, biggest lever if your photos ≠ Coralscapes)

Only worth it if you have (or can label) photos from your actual deployment setting.

1. Assemble a small set of **your** reef photos + hand-drawn masks (even 100-300),
   same class ids as Coralscapes (`classes.json`), Cityscapes-style layout, put them in
   their own `train/val` split under a folder.
2. Ask Claude to write `src/models/finetune_coralscapes.py`: load
   `EPFL-ECEO/coralscapes-vit-l-dpt`, fine-tune on **Coralscapes train + your set**
   (weight your set higher), cross-entropy with `ignore_index=0`, AdamW lr ~1e-5 for the
   backbone / 1e-4 for the head, batch 8-16, ~40-60 epochs, save to
   `models/checkpoints/algae_segmenter_ft/`.
3. Point `REPO_ID` (or add a `LOCAL_CHECKPOINT` path) in `algae_segmenter.py` at the
   fine-tuned weights, re-run the eval.

Budget: 1-3 hours on a big GPU. This is the only thing that makes it "perfect for *your*
data" rather than "good on Coralscapes val".

---

## Part E — sanity-check on real photos

```powershell
python src\inference\run_demo.py "path\to\one_of_your_reef_photos.jpg"   # --domain reef is default
```
Look at `data/processed/demo_output/<name>_annotated.jpg`:
- green = predicted algae, white = predicted bleaching
- terminal prints `algae cover X%` and `Y% of coral bleached`

Do this on 10-20 of your own photos. If the overlays land on the right regions, ship it.
If they're consistently off, that's the Part D signal.

---

## Part F — loose ends

- [ ] **Revoke the old HF token** `hf_qjq…` at https://huggingface.co/settings/tokens
      (it's in an old chat log). Use a fresh one.
- [ ] **Put the project in git** if it isn't yet:
      ```powershell
      git init; git add -A; git commit -m "reef segmentation pipeline"
      ```
- [ ] **ONNX export for deployment.** The reef model is a 443 MB - 1.2 GB ViT with a
      remote loader; it is NOT exported by `src/export/export_onnx.py`. Decide:
      - convert it (`torch.onnx` on `model.eval()`, bake in the rescale + ImageNet
        normalize from `preprocessor_config.json`, replicate the ÷16 resize client-side), or
      - ship `src/models/algae_classifier.py` (5 MB tile classifier, acc 0.82) for the
        no-Python target and keep the ViT for server/offline use.

---

## Quick reference

| thing | where |
|---|---|
| swap checkpoint / resolution / TTA | `src/models/algae_segmenter.py` (`REPO_ID`, `FULL_MAX_LONG_SIDE`, `_run_tta`) |
| reef inference entrypoint | `CoralDamagePipeline.run_reef()` in `src/pipeline/coral_damage_model.py` |
| reef eval | `eval_reef_segmenter()` in `src/eval/evaluate_all.py` |
| demo | `python src/inference/run_demo.py <img>` (`--domain reef` default) |
| class ids | `D:\coralscapes\coralscapes\classes.json`; groups in `algae_segmenter.py` (`CORAL_*_IDS`, `ALGAE_CLASS_ID`) |
| Coralscapes path | `CORALSCAPES_ROOT` env var or `_ROOT_CANDIDATES` lists in the data/eval scripts |
