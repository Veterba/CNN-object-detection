# CNN Object Detection

A from-scratch, single-scale YOLO-style object detector for car parts, built to
understand the machinery end-to-end: a small CNN backbone, a YOLO detection head,
a hand-written loss (target encoding), a hand-written decoder (NMS), and live
webcam inference. No pretrained backbone, no detection framework.

Detects 7 car-part classes: Bonet, Bumper, Door, Headlight, Mirror, Tailight, Windshield.

## Setup

The project runs in the bundled virtualenv at `./env` (git-ignored). Use it directly:

```bash
./env/bin/python train.py
```

or activate it once per shell:

```bash
source env/bin/activate
python train.py
```

> The pyenv-global `python3` does **not** have `pycocotools` installed — running
> there throws `ModuleNotFoundError: No module named 'pycocotools'` even though
> `requirements.txt` lists it. Always use the `env` venv.

Dependencies are pinned in `requirements.txt` (torch, torchvision, opencv, pycocotools, pillow, python-dotenv).

## Dataset

COCO-format export (Roboflow), split into `data/train` (668), `data/valid` (159),
`data/test` (123), each with its own `_annotations.coco.json`. Paths are set in
`config.py`.

## Usage

**Train** — fits the model, saves weights to `model.weights.pt`, prints per-epoch
loss and peak objectness, then evaluates:

```bash
./env/bin/python train.py
```

Watch `max_obj` per epoch, not just the loss: a collapsed detector (predicts "no
object everywhere") can still post a low loss. Healthy training drives `max_obj`
well off zero.

**Live detection** — opens the default camera, draws boxes + class/score, `q` to quit:

```bash
./env/bin/python capture.py
```

Point it at a car (it's a car-*parts* detector — your face yields nothing,
correctly). On macOS, `cv2.VideoCapture(0)` may grab a nearby iPhone via
Continuity Camera; switch to `VideoCapture(1)` in `capture.py` for the built-in webcam.

## How it works

- **Input** 128×128 RGB → 3 conv blocks (conv → BN → ReLU → maxpool) → 16×16 grid.
- **Head** a 1×1 conv outputs `3 anchors × (5 + 8 classes)` channels per cell:
  box offset (x, y, w, h), objectness, and class logits.
- **Loss** (`loss.py`) encodes each ground-truth box into the responsible
  (cell, anchor) by best-IoU, then sums coordinate / objectness / class terms.
- **Decode** (`decode.py`) is the exact inverse: un-transform offsets back to
  normalized boxes, threshold by `objectness × class_prob`, then NMS.

## Files

| File          | Role                                                      |
|---------------|-----------------------------------------------------------|
| `model.py`    | CNN backbone + detection head                             |
| `loss.py`     | YOLO loss and anchor priors (target encoding)             |
| `decode.py`   | Raw head output → detections (decoding + NMS)             |
| `dataset.py`  | COCO dataset wrapper + collate function                   |
| `train.py`    | Training loop and evaluation                              |
| `capture.py`  | Live webcam inference                                     |
| `config.py`   | Dataset paths and epoch count                             |

## Limitations

This is a learning project, not a production detector. From-scratch features,
~668 training images, no data augmentation, and a single 16×16 detection scale
cap the quality — expect confident-but-coarse boxes, scores roughly in the
0.2–0.6 range. The obvious next steps for real accuracy are data augmentation
and a pretrained backbone.

Both train and eval currently point at the same split via `config.py`; for an
honest generalization estimate, train on `data/train`, validate on `data/valid`,
and test once on `data/test`.
