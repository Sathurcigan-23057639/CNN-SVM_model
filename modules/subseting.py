import os
import shutil
import pandas as pd

def create_ordered_subset(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    image_column: str = "image",
    label_column: str = "level",
    samples_per_class: int = 500,
    seed: int = 42
):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    total_moved = 0
    log_entries = []

    # Explicit level order
    for level in [0, 1, 2, 3, 4]:
        level_df = df[df[label_column] == level]

        # If fewer than 500 images exist, take all
        n_samples = min(len(level_df), samples_per_class)
        sampled = level_df.sample(n=n_samples, random_state=seed)

        moved_count = 0
        for _, row in sampled.iterrows():
            img_name = row[image_column]
            img_level = row[label_column]

            # Try multiple extensions
            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src_path = os.path.join(image_dir, f"{img_name}{ext}")
                if os.path.exists(src_path):
                    dst_path = os.path.join(output_dir, f"{img_name}{ext}")
                    shutil.move(src_path, dst_path)
                    total_moved += 1
                    moved_count += 1
                    print(f"Moved: {img_name}{ext} | Level: {img_level}")
                    log_entries.append(f"Moved: {img_name}{ext} | Level: {img_level}")
                    found = True
                    break
            if not found:
                print(f"Missing file: {img_name}")
                log_entries.append(f"Missing file: {img_name}")

        print(f"Level {level}: moved {moved_count} images (requested {samples_per_class}, available {len(level_df)})")

    # Save full log to file
    log_path = os.path.join(output_dir, "moved_log.txt")
    with open(log_path, "w") as log_file:
        for entry in log_entries:
            log_file.write(entry + "\n")

    print(f"\n Finished. Moved {total_moved} images into {output_dir}")
    print(f" Full list of moved files saved to: {log_path}")


# Example usage
MAIN_PATH = "D:/Education/MSc/Active Assignments/Project/Model/KDR_Pre-processed"

create_ordered_subset(
    csv_path=os.path.join(MAIN_PATH, "traintestLabels15_trainLabels19.csv"),
    image_dir=MAIN_PATH,
    output_dir=os.path.join(MAIN_PATH, "subset"),
    samples_per_class=500
)