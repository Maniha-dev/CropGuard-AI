import os
import cv2

DATASET_DIR = "dataset"

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def check_split(split):

    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    print()
    print(f"Checking: {split}")

    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} not found")
        return

    if not os.path.exists(labels_dir):
        print(f"ERROR: {labels_dir} not found")
        return

    images = [
        file for file in os.listdir(images_dir)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    labels = [
        file for file in os.listdir(labels_dir)
        if file.lower().endswith(".txt")
    ]

    valid_images = 0
    invalid_images = 0
    missing_labels = 0

    for image_name in images:

        image_path = os.path.join(images_dir, image_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Invalid image: {image_name}")
            invalid_images += 1
            continue

        valid_images += 1

        base_name = os.path.splitext(image_name)[0]
        label_name = base_name + ".txt"

        if label_name not in labels:
            print(f"Missing label: {image_name}")
            missing_labels += 1

    print("------------------------------")
    print(f"Images:          {len(images)}")
    print(f"Labels:          {len(labels)}")
    print(f"Valid images:    {valid_images}")
    print(f"Invalid images:  {invalid_images}")
    print(f"Missing labels:  {missing_labels}")


print("======================================")
print("       CropGuard Dataset Check")
print("======================================")

for split in SPLITS:
    check_split(split)

print()
print("======================================")
print("       Dataset check completed")
print("======================================")