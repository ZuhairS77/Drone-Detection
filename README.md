# Drone Detection with YOLOv8

A small command-line project for training, evaluating, detecting, and tracking drones with Ultralytics YOLOv8. The repository contains the project code and dataset configuration; training data and generated model files are kept out of Git.

## Requirements

- Python 3.8 or newer
- pip
- A CUDA-capable GPU is recommended for training; CPU inference is supported

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Dataset

Download and unpack the YOLO Drone Detection Dataset. The default configuration in `coco.yaml` expects this layout:

```text
drone_dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

By default, `drone_dataset` is expected to be a sibling of this repository. Edit `path`, `train`, and `val` in `coco.yaml` if your dataset is stored elsewhere or has a different layout. Labels must use the YOLO detection format and class index `0` must represent `drone`.

The dataset is not included. Refer to the [YOLO Drone Detection Dataset on Kaggle](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) for its source and applicable terms.

## Usage

Run commands from the project directory. Ultralytics downloads `yolov8x.pt` automatically for the default training command if the weights are not already present.

Train:

```powershell
python drone_detection_yolov8x.py train --data coco.yaml --epochs 32 --batch 16 --imgsz 640
```

Evaluate weights:

```powershell
python drone_detection_yolov8x.py val --weights runs/detect/train/weights/best.pt --data coco.yaml
```

Detect objects in an image or video:

```powershell
python drone_detection_yolov8x.py predict --weights runs/detect/train/weights/best.pt --source path\to\image-or-video.mp4 --save
```

Track objects in a video or stream:

```powershell
python drone_detection_yolov8x.py track --weights runs/detect/train/weights/best.pt --source path\to\video.mp4 --show
```

Pass `--device 0` to select the first CUDA GPU or `--device cpu` to use the CPU. Run `python drone_detection_yolov8x.py --help` or append `--help` to a subcommand to see all options. Training and saved inference outputs are written under `runs/` by default.

The Jupyter notebook provides a lightweight example of calling the same CLI commands.

## Project files

- `drone_detection_yolov8x.py` — CLI for training, evaluation, prediction, and tracking
- `coco.yaml` — default dataset configuration
- `drone_detection_yolov8x.ipynb` — notebook usage example
- `requirements.txt` — Python dependency

## License

The project code is licensed under the MIT License; see [LICENSE](LICENSE). Dataset and third-party asset terms are separate.
