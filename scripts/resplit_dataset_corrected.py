"""
DO NOT USE - superseded by scripts/build_grouped_split.py (2026-10-01).

This script splits image-by-image (frames of one video land in different
splits) and copies into a target folder it never empties, so re-running it
duplicated 407 files across train/val/test. See docs/LEAKAGE_AUDIT.md.
"""
import sys
sys.exit("resplit_dataset_corrected.py is unsafe; use scripts/build_grouped_split.py")
"""
Corrected Dataset Resplitting Script
Uses ALL images from train/val/test directories and creates proper splits
"""
import shutil
import random
from pathlib import Path
import json

def resplit_dataset_all(source_dir="dataset_final", target_dir="dataset_final_resplit",
                       train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Resplit dataset using ALL images from train/val/test directories

    Args:
        source_dir: Source dataset directory
        target_dir: Target dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Source directory not found: {source_path}")
        return False

    # Create target directories
    for split in ['train', 'val', 'test']:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Collect ALL images from train/val/test directories
    all_image_files = []

    for split in ['train', 'val', 'test']:
        split_image_dir = source_path / split / 'images'
        split_label_dir = source_path / split / 'labels'

        if split_image_dir.exists():
            image_files = list(split_image_dir.glob("*.jpg")) + list(split_image_dir.glob("*.png"))
            print(f"Found {len(image_files)} images in {split}/")
            all_image_files.extend(image_files)

    if len(all_image_files) == 0:
        print("No image files found")
        return False

    print(f"Total images found: {len(all_image_files)}")

    # Remove duplicates (in case same image appears in multiple splits)
    unique_images = list(set(all_image_files))
    print(f"Unique images: {len(unique_images)}")

    # Shuffle files
    random.shuffle(unique_images)

    # Calculate split sizes
    total = len(unique_images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    print(f"\nNew split sizes:")
    print(f"  Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"  Val: {val_size} ({val_ratio*100:.1f}%)")
    print(f"  Test: {test_size} ({test_ratio*100:.1f}%)")

    # Split files
    train_files = unique_images[:train_size]
    val_files = unique_images[train_size:train_size + val_size]
    test_files = unique_images[train_size + val_size:]

    # Copy files to target directories
    def copy_files(files, split):
        copied = 0
        for img_file in files:
            # Copy image
            target_img = target_path / split / 'images' / img_file.name
            shutil.copy2(img_file, target_img)

            # Find and copy corresponding label
            label_file = None
            for source_split in ['train', 'val', 'test']:
                potential_label = source_path / source_split / 'labels' / f"{img_file.stem}.txt"
                if potential_label.exists():
                    label_file = potential_label
                    break

            if label_file:
                target_label = target_path / split / 'labels' / label_file.name
                shutil.copy2(label_file, target_label)
                copied += 1
            else:
                print(f"Warning: No label found for {img_file.name}")

        return copied

    print("\nCopying files...")
    train_copied = copy_files(train_files, 'train')
    val_copied = copy_files(val_files, 'val')
    test_copied = copy_files(test_files, 'test')

    print(f"\nFiles copied:")
    print(f"  Train: {train_copied} images with labels")
    print(f"  Val: {val_copied} images with labels")
    print(f"  Test: {test_copied} images with labels")

    # Create data.yaml file
    data_yaml_content = f"""path: {target_path.absolute()}
train: train/images
val: val/images
test: test/images

names:
  0: gallstone
"""

    yaml_path = target_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)

    print(f"\nCreated data.yaml at {yaml_path}")

    # Save split info
    split_info = {
        'total_images': total,
        'train': {'count': train_copied, 'ratio': train_ratio},
        'val': {'count': val_copied, 'ratio': val_ratio},
        'test': {'count': test_copied, 'ratio': test_ratio},
        'source_dir': str(source_path.absolute()),
        'target_dir': str(target_path.absolute()),
        'seed': seed
    }

    info_path = target_path / 'split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split info to {info_path}")

    return True

def main():
    print("=" * 60)
    print("CORRECTED DATASET RESPLITTING")
    print("=" * 60)

    # Resplit using ALL images
    success = resplit_dataset_all(
        source_dir="dataset_final",
        target_dir="dataset_final_resplit",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    if success:
        print("\n" + "=" * 60)
        print("RESPLITTING COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update your training scripts to use dataset_final_resplit")
        print("2. Re-run preprocessing on the new split")
        print("3. Train models with proper validation set")
    else:
        print("\nResplitting failed. Please check the error messages above.")

if __name__ == "__main__":
    main()