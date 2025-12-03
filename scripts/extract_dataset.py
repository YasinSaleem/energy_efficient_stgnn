#!/usr/bin/env python3
"""
Extract Household CSVs from Compressed Dataset

This script extracts individual household CSV files from the compressed
.tzst archives in the GoiEner Smart Meter Dataset.

The user can select which dataset to extract (imp-pre, imp-in, imp-post, or raw),
and the CSVs will be extracted to data/extracted/{dataset_name}/

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import os
import tarfile
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_DIR = "data/raw/7362094"
EXTRACTION_BASE_DIR = "data/extracted"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_available_datasets():
    """
    List all available .tzst datasets in the raw data directory.
    
    Returns:
        list: List of available .tzst files
    """
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Directory not found: {RAW_DATA_DIR}")
        return []
    
    tzst_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.tzst')]
    tzst_files.sort()
    
    return tzst_files


def display_dataset_menu(datasets):
    """
    Display available datasets and let user choose.
    
    Args:
        datasets (list): List of available dataset files
    
    Returns:
        str: Selected dataset filename, or None if user quits
    """
    print("\n" + "="*70)
    print("  AVAILABLE DATASETS FOR EXTRACTION")
    print("="*70)
    
    for idx, dataset in enumerate(datasets, 1):
        # Get file size
        file_path = os.path.join(RAW_DATA_DIR, dataset)
        file_size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
        print(f"  {idx}. {dataset:20s} ({file_size:.2f} GB)")
    
    print("="*70)
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected = datasets[choice_num - 1]
                print(f"\n✓ Selected: {selected}")
                return selected
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q' to quit")


def extract_dataset(dataset_file):
    """
    Extract all household CSVs from a compressed archive.
    
    Args:
        dataset_file (str): Name of the .tzst file to extract
    
    Returns:
        dict: Statistics about the extraction
    """
    # Construct paths
    input_path = os.path.join(RAW_DATA_DIR, dataset_file)
    
    # Determine output directory based on dataset name
    dataset_name = dataset_file.replace('.tzst', '')
    output_dir = os.path.join(EXTRACTION_BASE_DIR, dataset_name)
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING DATASET")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already extracted
    existing_files = list(Path(output_dir).rglob("*.csv"))
    if len(existing_files) > 0:
        print(f"\n⚠️  Warning: Found {len(existing_files)} existing CSV files in output directory")
        response = input("Do you want to re-extract (will overwrite)? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Extraction cancelled")
            return None
    
    # Extract archive
    print("\nExtracting compressed archive...")
    
    csv_count = 0
    total_size = 0
    
    try:
        with open(input_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    
                    for member in tqdm(tar, desc="Extracting files"):
                        if member.name.endswith('.csv'):
                            # Extract to output directory
                            tar.extract(member, path=output_dir)
                            csv_count += 1
                            total_size += member.size
    
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        return None
    
    # Calculate statistics
    stats = {
        'dataset_name': dataset_name,
        'csv_count': csv_count,
        'total_size_mb': total_size / (1024**2),
        'output_dir': output_dir
    }
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETED")
    print(f"{'='*70}")
    print(f"Extracted files:     {csv_count:,}")
    print(f"Total size:          {stats['total_size_mb']:.2f} MB")
    print(f"Output directory:    {output_dir}")
    
    # Verify extraction
    extracted_files = list(Path(output_dir).rglob("*.csv"))
    print(f"Verified CSV files:  {len(extracted_files):,}")
    
    if len(extracted_files) != csv_count:
        print(f"\n⚠️  Warning: Mismatch between extracted ({csv_count}) and verified ({len(extracted_files)}) files")
    
    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  DATASET EXTRACTION TOOL")
    print("="*70)
    
    try:
        # Step 1: List available datasets
        available_datasets = list_available_datasets()
        
        if not available_datasets:
            print(f"\n❌ No .tzst datasets found in {RAW_DATA_DIR}")
            return
        
        # Step 2: Let user select dataset
        selected_dataset = display_dataset_menu(available_datasets)
        
        if selected_dataset is None:
            print("\n👋 Exiting...")
            return
        
        # Step 3: Extract the selected dataset
        stats = extract_dataset(selected_dataset)
        
        if stats:
            print(f"\n{'='*70}")
            print(f"✓ EXTRACTION SUCCESSFUL!")
            print(f"{'='*70}")
            print(f"\nYou can now use the extracted files from:")
            print(f"  {stats['output_dir']}/")
            print(f"\nExample usage:")
            print(f"  python scripts/split_pre_covid_dataset.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*70}")
        raise


if __name__ == "__main__":
    main()
