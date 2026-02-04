import os
import argparse
from datetime import datetime
from typing import Dict, Any, List

# ===============================
# GPU CONTROL (BEFORE TF)
# ===============================
def set_gpu(gpu_num: str | None):
    if gpu_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        print(f"[INFO] Using GPU device: {gpu_num}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] Running on CPU")


# ===============================
# ARGPARSE (EARLY)
# ===============================
parser = argparse.ArgumentParser(
    description="Face Recognition with PostgreSQL (DeepFace + pgvector ANN)"
)
parser.add_argument("--db_path", required=True, help="Folder with images to REGISTER")
parser.add_argument("--input_img", required=True, help="Folder with QUERY images")
parser.add_argument("--result_path", required=True, help="Output result file")
parser.add_argument("--gpu_num", default=None, help="GPU number (e.g. 0)")

args = parser.parse_args()
set_gpu(args.gpu_num)

# ===============================
# SAFE IMPORTS
# ===============================
import tensorflow as tf
from deepface import DeepFace
import pandas as pd

# ===============================
# CONFIG
# ===============================
DF_RECOGNITION_CONFIG: Dict[str, Any] = dict(
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True,
    l2_normalize=True,
)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ===============================
# UTILS
# ===============================
def get_image_list(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

def extract_identity(row: pd.Series) -> str:
    """
    Safely extract identity from ANN search result row
    (Postgres / pgvector compatible)
    """
    for key in ["identity", "img_name", "identity_id", "name"]:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return "UNKNOWN"



# ===============================
# MODEL PRELOAD
# ===============================
def load_model():
    print("[INFO] Loading ArcFace model...")
    DeepFace.build_model("ArcFace")
    print("[INFO] Model loaded")


# ===============================
# REGISTRATION STEP
# ===============================
def register_database_images(db_path: str):
    print("[INFO] Registering images into PostgreSQL...")

    images = get_image_list(db_path)
    if not images:
        raise RuntimeError("No images found in db_path folder")

    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        try:
            print(f"[REGISTER] {img_name}")
            DeepFace.register(
                img=img,
                img_name=img_name,
                **DF_RECOGNITION_CONFIG
            )
        except Exception as e:
            print(f"[WARN] Failed to register {img}: {e}")

    print("[INFO] Registration completed")


# ===============================
# BUILD ANN INDEX
# ===============================
def build_ann_index():
    print("[INFO] Building ANN index (pgvector)...")
    DeepFace.build_index(
        model_name="ArcFace",
        detector_backend="retinaface",
        align=True,
        l2_normalize=True,
        database_type="postgres",
    )
    print("[INFO] ANN index ready")


# ===============================
# SEARCH STEP (ANN)
# ===============================
def run_search(input_img: str, result_path: str):
    images = get_image_list(input_img)
    if not images:
        raise RuntimeError("No images found in input_img folder")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Face Recognition Results (PostgreSQL ANN)\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

        for img_path in images:
            img_name = os.path.basename(img_path)
            print(f"[SEARCH] {img_name}")

            try:
                dfs: List[pd.DataFrame] = DeepFace.search(
                    img=img_path,
                    search_method="ann",
                    **DF_RECOGNITION_CONFIG
                )

                if dfs and not dfs[0].empty:
                    best = dfs[0].iloc[0]

                    identity = extract_identity(best)
                    distance = best.get("distance", None)
                    threshold = best.get("threshold", None)

                    f.write(f"[FOUND] {img_name}\n")
                    f.write(f"  Identity  : {identity}\n")

                    if distance is not None:
                        f.write(f"  Distance  : {distance:.6f}\n")
                    if threshold is not None:
                        f.write(f"  Threshold : {threshold}\n")

                    f.write("\n")
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
    print("[INFO] TensorFlow:", tf.__version__)
    print("[INFO] GPUs:", tf.config.list_physical_devices("GPU"))

    load_model()
    register_database_images(args.db_path)
    build_ann_index()
    run_search(args.input_img, args.result_path)


if __name__ == "__main__":
    main()
