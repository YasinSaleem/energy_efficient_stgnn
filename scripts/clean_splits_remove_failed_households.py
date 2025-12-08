#!/usr/bin/env python3
"""
Clean Splits - Remove Failed Households

This script removes all households that failed validation from all split folders.
It reads the list of failed households from the validation report and deletes
their corresponding CSV files from all split directories.

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FAILED_HOUSEHOLDS_FILE = PROJECT_ROOT / "data" / "validation" / "failed_households.txt"

# Split folders to clean
SPLIT_FOLDERS = [
    SPLITS_DIR / "train",
    SPLITS_DIR / "val",
    SPLITS_DIR / "test",
    SPLITS_DIR / "continual" / "CL_1",
    SPLITS_DIR / "continual" / "CL_2",
    SPLITS_DIR / "continual" / "CL_3",
    SPLITS_DIR / "continual" / "CL_4",
]

SPLIT_NAMES = ["train", "val", "test", "cl_1", "cl_2", "cl_3", "cl_4"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_failed_households(failed_file):
    """
    Load the list of failed household IDs from the validation report.

    Args:
        failed_file (Path): Path to the failed_households.txt file

    Returns:
        set: Set of failed household IDs
    """
    if not failed_file.exists():
        raise FileNotFoundError(f"Failed households file not found: {failed_file}")

    failed_ids = set()

    with open(failed_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Skip lines that start with whitespace followed by [ (error details)
            if line.startswith('  [') or line.startswith('    -'):
                continue

            # This is a household ID
            failed_ids.add(line)

    return failed_ids


def count_files_in_folder(folder):
    """
    Count the number of CSV files in a folder.

    Args:
        folder (Path): Path to the folder

    Returns:
        int: Number of CSV files
    """
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.csv")))


def get_household_id_from_filename(filepath):
    """
    Extract household ID from a CSV filename.

    Args:
        filepath (Path): Path to the CSV file

    Returns:
        str: Household ID (filename without .csv extension)
    """
    return filepath.stem


def clean_split_folder(folder, failed_ids, dry_run=False):
    """
    Remove failed household CSVs from a split folder.

    Args:
        folder (Path): Path to the split folder
        failed_ids (set): Set of failed household IDs
        dry_run (bool): If True, only show what would be deleted

    Returns:
        dict: Statistics about the cleaning operation
    """
    if not folder.exists():
        return {
            'before': 0,
            'deleted': 0,
            'after': 0,
            'errors': []
        }

    # Get all CSV files
    csv_files = list(folder.glob("*.csv"))
    before_count = len(csv_files)

    deleted_count = 0
    errors = []

    # Process each file
    for csv_file in csv_files:
        household_id = get_household_id_from_filename(csv_file)

        if household_id in failed_ids:
            if dry_run:
                # Just count, don't delete
                deleted_count += 1
            else:
                # Actually delete the file
                try:
                    csv_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Failed to delete {csv_file.name}: {e}")

    after_count = before_count - deleted_count

    return {
        'before': before_count,
        'deleted': deleted_count,
        'after': after_count,
        'errors': errors
    }


def print_summary(stats, total_failed, dry_run=False):
    """
    Print a summary of the cleaning operation.

    Args:
        stats (dict): Dictionary of statistics per split
        total_failed (int): Total number of failed households
        dry_run (bool): Whether this was a dry run
    """
    mode = "[DRY RUN] " if dry_run else ""

    print("\n" + "="*70)
    print(f"{mode}CLEANING COMPLETE".center(70))
    print("="*70)

    # Calculate totals
    total_before = sum(s['before'] for s in stats.values())
    total_deleted = sum(s['deleted'] for s in stats.values())
    total_after = sum(s['after'] for s in stats.values())

    # Calculate unique households (from train split)
    train_before = stats.get('train', {}).get('before', 0)
    train_after = stats.get('train', {}).get('after', 0)

    print(f"\nFailed households to remove:  {total_failed:,}")
    print(f"Files deleted across splits:  {total_deleted:,}")
    print(f"Valid households remaining:   {train_after:,}")

    print(f"\nPer-Split Statistics:")
    print("-" * 70)
    print(f"{'Split':<10} {'Before':<12} {'Deleted':<12} {'After':<12}")
    print("-" * 70)

    for split_name in SPLIT_NAMES:
        s = stats.get(split_name, {'before': 0, 'deleted': 0, 'after': 0})
        before = f"{s['before']:,}"
        deleted = f"{s['deleted']:,}"
        after = f"{s['after']:,}"
        print(f"{split_name:<10} {before:<12} {deleted:<12} {after:<12}")

    print("-" * 70)
    print(f"{'TOTAL':<10} {total_before:<12,} {total_deleted:<12,} {total_after:<12,}")
    print("-" * 70)

    # Check for errors
    total_errors = sum(len(s.get('errors', [])) for s in stats.values())
    if total_errors > 0:
        print(f"\n⚠️  Encountered {total_errors} errors during cleaning")
        for split_name, s in stats.items():
            if s.get('errors'):
                print(f"\n{split_name}:")
                for error in s['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(s['errors']) > 5:
                    print(f"  ... and {len(s['errors']) - 5} more")

    # Verify consistency
    unique_counts = {split_name: stats[split_name]['after']
                     for split_name in SPLIT_NAMES if split_name in stats}

    if len(set(unique_counts.values())) == 1:
        print(f"\n✓ All splits are now consistent with {train_after:,} households each")
    else:
        print(f"\n⚠️  Warning: Split counts are inconsistent!")
        for split_name, count in unique_counts.items():
            print(f"  {split_name}: {count:,}")

    print("="*70)

    if dry_run:
        print("\n💡 This was a dry run. No files were actually deleted.")
        print("   Run without --dry-run to perform the actual deletion.")
    else:
        print("\n✓ Cleaning completed successfully!")

    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Remove failed households from all split folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted (recommended first)
  python clean_splits_remove_failed_households.py --dry-run

  # Actually delete the files
  python clean_splits_remove_failed_households.py
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting files'
    )

    parser.add_argument(
        '--failed-file',
        type=Path,
        default=FAILED_HOUSEHOLDS_FILE,
        help=f'Path to failed households file (default: {FAILED_HOUSEHOLDS_FILE})'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CLEAN SPLITS - REMOVE FAILED HOUSEHOLDS".center(70))
    print("="*70)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted")

    try:
        # Step 1: Load failed households
        print(f"\nLoading failed households from: {args.failed_file}")
        failed_ids = load_failed_households(args.failed_file)
        print(f"✓ Loaded {len(failed_ids):,} failed household IDs")

        # Step 2: Validate split folders exist
        print(f"\nValidating split folders...")
        missing_folders = [f for f in SPLIT_FOLDERS if not f.exists()]
        if missing_folders:
            print(f"\n⚠️  Warning: The following folders do not exist:")
            for folder in missing_folders:
                print(f"  - {folder}")
            response = input("\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return

        # Step 3: Count files before cleaning
        print(f"\nCounting files in split folders...")
        initial_counts = {}
        for folder, name in zip(SPLIT_FOLDERS, SPLIT_NAMES):
            count = count_files_in_folder(folder)
            initial_counts[name] = count
            print(f"  {name:<10} - {count:,} files")

        # Step 4: Clean each folder
        print(f"\nCleaning split folders...")
        stats = {}

        for folder, split_name in tqdm(
            list(zip(SPLIT_FOLDERS, SPLIT_NAMES)),
            desc="Processing splits"
        ):
            folder_stats = clean_split_folder(folder, failed_ids, dry_run=args.dry_run)
            stats[split_name] = folder_stats

        # Step 5: Print summary
        print_summary(stats, len(failed_ids), dry_run=args.dry_run)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
