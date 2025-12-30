import os
import shutil
import pandas as pd

def create_subset(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    image_column: str = "image",
    label_column: str = "level",
    seed: int = 42
):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Show counts per class first
    print("\nImage counts per class:")
    class_counts = {}
    for level in [0, 1, 2, 3, 4]:
        count = len(df[df[label_column] == level])
        class_counts[level] = count
        print(f"Class {level}: {count} images")

    # Select number of samples per class
    while True:
        try:
            samples_per_class = int(input("\nHow many images want to pick: "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    total_moved = 0
    log_entries = []
    moved_records = []

    # Process each class
    for level in [0, 1, 2, 3, 4]:
        level_df = df[df[label_column] == level]
        available = class_counts[level]

        # Take the minimum of available images if higher than requested
        n_samples = min(available, samples_per_class)
        sampled = level_df.sample(n=n_samples, random_state=seed)

        moved_count = 0
        for _, row in sampled.iterrows():
            img_name = row[image_column]
            img_level = row[label_column]

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
                    moved_records.append({
                        image_column: f"{img_name}{ext}",
                        label_column: img_level
                    })
                    found = True
                    break
            if not found:
                print(f"Missing file: {img_name}")
                log_entries.append(f"Missing file: {img_name}")

        print(f"Level {level}: moved {moved_count} images (requested {samples_per_class}, available {available})")

    # Save log
    log_path = os.path.join(output_dir, "moved_log.txt")
    with open(log_path, "w") as log_file:
        for entry in log_entries:
            log_file.write(entry + "\n")

    # Save new CSV
    new_csv_path = os.path.join(output_dir, "subset_labels.csv")
    if moved_records:
        df_moved = pd.DataFrame(moved_records)
        df_moved.to_csv(new_csv_path, index=False)
        print(f"New CSV for moved images saved to: {new_csv_path}")
    else:
        print("No images were moved; CSV not created.")

    print(f"\nFinished. Total images moved: {total_moved}")


# Execution of the function
MAIN_PATH = "D:/Education/MSc/Active Assignments/Project/Model/KDR_Pre-processed"

create_subset(
    csv_path=os.path.join(MAIN_PATH, "traintestLabels15_trainLabels19.csv"),
    image_dir=MAIN_PATH,
    output_dir=os.path.join(MAIN_PATH, "subset")
)