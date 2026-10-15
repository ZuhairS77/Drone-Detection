"""Train, evaluate, and run YOLOv8 drone detection."""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "coco.yaml"


def add_data_options(parser):
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Dataset YAML file (default: coco.yaml in the project directory).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", help="Inference device, for example '0' or 'cpu'.")


def add_output_options(parser, default_name):
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Directory in which to save run outputs.",
    )
    parser.add_argument("--name", default=default_name, help="Name of this run.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and run a YOLOv8 drone detection model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a model.")
    add_data_options(train)
    train.add_argument(
        "--model",
        default="yolov8x.pt",
        help="Initial model weights or model configuration (default: yolov8x.pt).",
    )
    train.add_argument("--epochs", type=int, default=32, help="Number of training epochs.")
    train.add_argument("--batch", type=int, default=16, help="Training batch size.")
    add_output_options(train, "train")

    validate = commands.add_parser("val", help="Evaluate model weights.")
    add_data_options(validate)
    validate.add_argument("--weights", required=True, help="Model weights to evaluate.")
    add_output_options(validate, "val")

    for command, help_text in (
        ("predict", "Run object detection on an image, video, or stream."),
        ("track", "Track detected objects in a video or stream."),
    ):
        inference = commands.add_parser(command, help=help_text)
        inference.add_argument("--weights", required=True, help="Model weights to use.")
        inference.add_argument(
            "--source", required=True, help="Image, video, webcam index, or stream URL."
        )
        inference.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
        inference.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
        inference.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
        inference.add_argument("--device", help="Inference device, for example '0' or 'cpu'.")
        inference.add_argument(
            "--show", action="store_true", help="Display inference results in a window."
        )
        inference.add_argument(
            "--save", action="store_true", help="Save annotated inference results."
        )
        add_output_options(inference, command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("train", "val"):
        data_path = args.data.expanduser().resolve()
        if not data_path.is_file():
            parser.error(f"Dataset YAML file does not exist: {data_path}")
        args.data = data_path

    if args.command == "train":
        model = YOLO(args.model)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(args.project),
            name=args.name,
            device=args.device,
        )
    elif args.command == "val":
        model = YOLO(args.weights)
        model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            project=str(args.project),
            name=args.name,
            device=args.device,
        )
    else:
        model = YOLO(args.weights)
        inference_options = {
            "source": args.source,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "show": args.show,
            "save": args.save,
            "project": str(args.project),
            "name": args.name,
        }
        if args.device is not None:
            inference_options["device"] = args.device

        if args.command == "track":
            model.track(**inference_options)
        else:
            model.predict(**inference_options)


if __name__ == "__main__":
    main()
