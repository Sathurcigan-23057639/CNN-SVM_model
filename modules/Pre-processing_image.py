import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ImagePreprocessing:
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.encoder = LabelEncoder()

    def load_and_resize(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")
        return cv2.resize(image, self.image_size)

    def normalize(self, image):
        return image / 255.0

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def reduce_noise(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def preprocess_image(self, image_path):
        image = self.load_and_resize(image_path)
        image = self.enhance_contrast(image)
        image = self.reduce_noise(image)
        image = self.normalize(image)
        return image

    def save_image(self, image, output_path):
        image_to_save = (image * 255).astype(np.uint8)
        cv2.imwrite(output_path, image_to_save)

    def augment_image(self, image):
        flipped = cv2.flip(image, 1)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        return [image, flipped, rotated]

    def encode_labels(self, labels):
        return self.encoder.fit_transform(labels)

    def split_data(self, images, labels, test_size=0.2):
        if len(images) == 0:
            raise ValueError("No images were loaded. Check your image directories and label CSVs.")
        return train_test_split(images, labels, test_size=test_size, stratify=labels)

    def load_labels_from_csvs(self, csv_paths, severity_map=None):
        label_dict = {}
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"Skipping missing CSV: {csv_path}")
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
        return label_dict


    def batch_preprocess_from_csvs(self, image_dirs, csv_paths, severity_map=None, save_dir=None):
        label_dict = self.load_labels_from_csvs(csv_paths, severity_map)
        images, labels = [], []

        for image_dir in image_dirs:
            if not os.path.exists(image_dir):
                print(f"Skipping missing image folder: {image_dir}")
                continue
            for filename in os.listdir(image_dir):
                if filename in label_dict:
                    path = os.path.join(image_dir, filename)
                    try:
                        image = self.preprocess_image(path)
                        images.append(image)
                        labels.append(label_dict[filename])
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, filename)
                            self.save_image(image, save_path)
                    except MemoryError:
                        print(f"Skipping {filename}: MemoryError during processing.")
                    except cv2.error as e:
                        print(f"Skipping {filename}: OpenCV error → {e}")
                    except Exception as e:
                        print(f"Skipping {filename}: {e}")
                else:
                    print(f"Label not found for {filename}, skipping.")

        return self.split_data(np.array(images), self.encode_labels(labels))



if __name__ == "__main__":
    processor = ImagePreprocessing()

    image_dirs = [
        "../../Kaggle_Diabetic_Retinopathy/resized_traintest15_train19"
    ]

    csv_paths = [
        "../../Kaggle_Diabetic_Retinopathy/labels/testLabels15.csv",
        "../../Kaggle_Diabetic_Retinopathy/labels/trainLabels15.csv",
        "../../Kaggle_Diabetic_Retinopathy/labels/trainLabels19.csv",
        "../../Kaggle_Diabetic_Retinopathy/labels/traintestLabels15_validation.csv"
    ]

    save_dir = "../../KDR_Pre-processed"

    severity_map = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }

    X_train, X_test, y_train, y_test = processor.batch_preprocess_from_csvs(
        image_dirs=image_dirs,
        csv_paths=csv_paths,
        severity_map=severity_map,
        save_dir=save_dir
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))