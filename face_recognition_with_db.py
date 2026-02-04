import os
import argparse
from datetime import datetime
from typing import Dict, Any

# ===============================
# GPU CONTROL (BEFORE TENSORFLOW)
# ===============================
def set_gpu(gpu_num: str | None):
    """
    Control GPU usage via CUDA_VISIBLE_DEVICES
    """
    if gpu_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        print(f"[INFO] Using GPU device: {gpu_num}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] Running on CPU")


# ===============================
# ARGPARSE (early for GPU control)
# ===============================
parser = argparse.ArgumentParser(
    description="Face Recognition with Local Database (DeepFace ArcFace)"
)
parser.add_argument("--db_path", required=True, help="Path to face database folder")
parser.add_argument("--input_img", required=True, help="Folder with query images")
parser.add_argument("--result_path", required=True, help="Output result .txt file")
parser.add_argument("--gpu_num", default=None, help="GPU number (e.g. 0). Omit for CPU")

args = parser.parse_args()
set_gpu(args.gpu_num)

# ===============================
# IMPORT TF & DEEPFACE
# ===============================
import tensorflow as tf
from deepface import DeepFace

# ===============================
# Face Recognition CONFIG
# ===============================
DF_RECOGNITION_CONFIG: Dict[str, Any] = dict(
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True
)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ===============================
# MODEL LOADING
# ===============================
def load_model():
    """
    Preload ArcFace model into memory (important for speed)
    """
    print("[INFO] Loading ArcFace model...")
    DeepFace.build_model("ArcFace")
    print("[INFO] Model loaded")


# ===============================
# UTILS
# ===============================
def get_image_list(folder: str):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]


# ===============================
# CORE LOGIC
# ===============================
def run_face_recognition(db_path: str, input_img: str, result_path: str):
    if not os.path.isdir(db_path):
        raise RuntimeError(f"Database folder not found: {db_path}")

    if not os.path.isdir(input_img):
        raise RuntimeError(f"Input image folder not found: {input_img}")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    images = get_image_list(input_img)
    if not images:
        raise RuntimeError("No images found in input_img folder")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Face Recognition Results\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

        for img_path in images:
            img_name = os.path.basename(img_path)
            print(f"[INFO] Processing: {img_name}")

            try:
                result: Any = DeepFace.find(
                    img_path=img_path,
                    db_path=db_path,
                    **DF_RECOGNITION_CONFIG
                )

                if len(result) > 0 and not result[0].empty:
                    best = result[0].iloc[0]

                    f.write(f"[FOUND] {img_name}\n")
                    f.write(f"  Match      : {best['identity']}\n")
                    f.write(f"  Distance   : {best['distance']:.6f}\n")
                    f.write(f"  Threshold  : {best['threshold']}\n")
                    f.write(f"  Confidence : {best['confidence']:.2f}\n\n")
                else:
                    f.write(f"[NOT FOUND] {img_name}\n\n")

            except Exception as e:
                f.write(f"[ERROR] {img_name}\n")
                f.write(f"  Reason: {str(e)}\n\n")

    print(f"[DONE] Results saved to: {result_path}")


# ===============================
# ENTRY POINT
# ===============================
def main():
    print("[INFO] TensorFlow version:", tf.__version__)
    print("[INFO] GPU available:", tf.config.list_physical_devices("GPU"))

    load_model()

    run_face_recognition(
        db_path=args.db_path,
        input_img=args.input_img,
        result_path=args.result_path
    )


if __name__ == "__main__":
    main()
