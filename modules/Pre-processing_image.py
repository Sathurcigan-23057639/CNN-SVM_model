import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ImagePreprocessing:
    def __init__(self, image_size=(512, 512), max_images=None, log_skipped=True):
        self.image_size = image_size
        self.encoder = LabelEncoder()
        self.max_images = max_images
        self.log_skipped = log_skipped

    def load_and_resize(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"The Image is not found '{image_path}' in this path.")
            return cv2.resize(image, self.image_size)
        except cv2.error as e:
            raise RuntimeError(f"OpenCV error while loading {image_path}: {e}")
        except MemoryError:
            raise RuntimeError(f"Memory error while loading {image_path}")

    @staticmethod
    def normalize(image):
        return (image / 255.0).astype(np.float32)

    @staticmethod
    def enhance_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def reduce_noise(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def preprocess_image(self, image_path):
        image = self.load_and_resize(image_path)
        image = self.enhance_contrast(image)
        image = self.reduce_noise(image)
        image = self.normalize(image)
        return image

    @staticmethod
    def save_image(image, output_path):
        image_to_save = (image * 255).astype(np.uint8)
        cv2.imwrite(output_path, image_to_save)

    def encode_labels(self, labels):
        return self.encoder.fit_transform(labels)

    def _process_images_streaming(self, image_dirs, label_dict, save_dir):
        image_paths, labels = [], []
        count = 0

        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                print(f"The selected directory is not found: {image_dir}")
                continue

            for filename in os.listdir(image_dir):
                if self.max_images and count >= self.max_images:
                    break

                filename = filename.lower()
                if not filename.endswith((".jpg", ".jpeg", ".png", ".tif")):
                    continue

                if filename in label_dict:
                    path = os.path.join(image_dir, filename)
                    try:
                        image = self.preprocess_image(path)
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, filename)
                            self.save_image(image, save_path)
                            image_paths.append(save_path)
                        else:
                            image_paths.append(path)
                        labels.append(label_dict[filename])
                        count += 1
                    except Exception as e:
                        print(f"Skipping {filename}: {e}")
                        if self.log_skipped:
                            with open("skipped_images.txt", "a") as log:
                                log.write(f"{filename}: {e}\n")
                else:
                    print(f"Label not found for {filename}, skipping.")
                    if self.log_skipped:
                        with open("skipped_images.txt", "a") as log:
                            log.write(f"{filename}: label not found\n")

        if len(image_paths) == 0 or len(labels) == 0:
            raise ValueError("No valid Messidor images or labels found. Check your XLS files and image folders.")

        return train_test_split(image_paths, self.encode_labels(labels), test_size=0.2, stratify=labels)

    def preprocess_kaggle_dataset(self, image_dirs, csv_paths, severity_map=None, save_dir=None):
        label_dict = {}
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"CSV file is not found in '{csv_path}' this path.")
                continue
            try:
                df = pd.read_csv(csv_path)
                df.columns = [col.lower().strip() for col in df.columns]
                if "image" not in df.columns or "level" not in df.columns:
                    print(f"Skipping {csv_path}: missing 'image' or 'level' column.")
                    continue
                for _, row in df.iterrows():
                    filename = str(row["image"]).strip() + ".jpg"
                    grade = int(row["level"])
                    label_dict[filename] = severity_map.get(grade, grade) if severity_map else grade
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        return self._process_images_streaming(image_dirs, label_dict, save_dir)

    def preprocess_messidor_dataset(self, image_dirs, xls_paths, severity_map=None, save_dir=None):
        label_dict = {}

        for xls_path in xls_paths:
            if not os.path.exists(xls_path):
                print(f"The selected XLS file is not found in '{xls_path}' this path.")
                continue
            try:
                df = pd.read_excel(xls_path)
                df.columns = [col.lower().strip() for col in df.columns]
                if "image name" not in df.columns or "retinopathy grade" not in df.columns:
                    print(f"Skipping {xls_path}: missing 'image name' or 'retinopathy grade' column.")
                    continue
                for _, row in df.iterrows():
                    filename = str(row["image name"]).strip().lower()
                    filename += ".tif" if not filename.endswith(".tif") else ""
                    label_dict[filename] = severity_map.get(int(row["retinopathy grade"]), row["retinopathy grade"])
            except Exception as e:
                print(f"Error reading {xls_path}: {e}")

        return self._process_images_streaming(image_dirs, label_dict, save_dir)



if __name__ == "__main__":
    processor = ImagePreprocessing(max_images=None, log_skipped=True)

    # Kaggle
    kaggle_image_dirs = ["../../Kaggle_Diabetic_Retinopathy/resized_traintest15_train19"]
    kaggle_csv_paths = ["../../Kaggle_Diabetic_Retinopathy/labels/traintestLabels15_trainLabels19.csv"]
    kaggle_severity_map = {
        0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"
    }
    kaggle_save_dir = "../../KDR_Pre-processed"

    print("Kaggle DR Image Pre-processing Started.")

    X_train, X_test, y_train, y_test = processor.preprocess_kaggle_dataset(
        image_dirs=kaggle_image_dirs,
        csv_paths=kaggle_csv_paths,
        severity_map=kaggle_severity_map,
        save_dir=kaggle_save_dir
    )
    print("Kaggle training samples:", len(X_train))
    print("Kaggle test samples:", len(X_test))

    # Messidor
    messidor_image_dirs = [
        "../../Messidor/Base11",
        "../../Messidor/Base21",
        "../../Messidor/Base31"
    ]
    messidor_csv_paths = [
        "../../Messidor/Base11/Annotation_Base11.xls",
        "../../Messidor/Base21/Annotation_Base21.xls",
        "../../Messidor/Base31/Annotation_Base31.xls"
    ]
    messidor_severity_map = {
        0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe"
    }
    messidor_save_dir = "../../MS_Pre-processed"

    print("Messidor Image Pre-processing Started.")

    X_train, X_test, Y_train, Y_test = processor.preprocess_messidor_dataset(
        image_dirs=messidor_image_dirs,
        xls_paths=messidor_csv_paths,
        severity_map=messidor_severity_map,
        save_dir=messidor_save_dir
    )

    print("Messidor training samples:", len(X_train))
    print("Messidor test samples:", len(X_test))
