"""
Merge Hinglish prompt injection datasets from CSV and Excel formats.
Standardizes column names, removes duplicates, and outputs consolidated training data.
"""

import pandas as pd
import os
from pathlib import Path


def read_datasets(csv_path=None, excel_path=None):
    """
    Read datasets from CSV and Excel formats.
    
    Args:
        csv_path: Path to custom dataset CSV file
        excel_path: Path to Srinivasan dataset Excel file
    
    Returns:
        list: List of dataframes
    """
    dataframes = []
    
    # Read CSV dataset
    if csv_path and os.path.exists(csv_path):
        print(f"Reading CSV dataset from: {csv_path}")
        df_csv = pd.read_csv(csv_path)
        print(f"  - Loaded {len(df_csv)} samples")
        print(f"  - Columns: {list(df_csv.columns)}")
        dataframes.append(df_csv)
    else:
        print(f"Warning: CSV dataset not found at {csv_path}")
    
    # Read Excel dataset
    if excel_path and os.path.exists(excel_path):
        print(f"Reading Excel dataset from: {excel_path}")
        df_excel = pd.read_excel(excel_path)
        print(f"  - Loaded {len(df_excel)} samples")
        print(f"  - Columns: {list(df_excel.columns)}")
        dataframes.append(df_excel)
    else:
        print(f"Warning: Excel dataset not found at {excel_path}")
    
    if not dataframes:
        raise ValueError("No datasets found. Please provide valid file paths.")
    
    return dataframes


def standardize_columns(dataframes):
    """
    Standardize column names across datasets.
    Maps various column names to 'text' and 'label'.
    
    Args:
        dataframes: List of dataframes
    
    Returns:
        list: List of dataframes with standardized columns
    """
    # Common column name variations
    text_aliases = ['text', 'prompt', 'input', 'sentence', 'query', 'content', 'data']
    label_aliases = ['label', 'class', 'target', 'injection', 'is_injection', 'category']
    
    standardized = []
    
    for df in dataframes:
        df_copy = df.copy()
        columns_orig = list(df_copy.columns)
        columns_lower = [col.lower() for col in columns_orig]
        
        # Find text column with flexible matching
        text_col = None
        for alias in text_aliases:
            for i, col_lower in enumerate(columns_lower):
                # Exact match or substring match
                if col_lower == alias or alias in col_lower:
                    text_col = columns_orig[i]
                    break
            if text_col:
                break
        
        # Find label column with flexible matching
        label_col = None
        for alias in label_aliases:
            for i, col_lower in enumerate(columns_lower):
                # Exact match or substring match
                if col_lower == alias or alias in col_lower:
                    label_col = columns_orig[i]
                    break
            if label_col:
                break
        
        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not identify text and label columns in dataset. "
                f"Found columns: {list(df_copy.columns)}\n"
                f"Expected: text column (aliases: {text_aliases}) and "
                f"label column (aliases: {label_aliases})"
            )
        
        # Rename columns
        df_copy = df_copy.rename(columns={text_col: 'text', label_col: 'label'})
        df_copy = df_copy[['text', 'label']]
        
        # Clean whitespace
        df_copy['text'] = df_copy['text'].astype(str).str.strip()
        
        standardized.append(df_copy)
        print(f"Standardized dataframe: {len(df_copy)} samples (from columns: {text_col}, {label_col})")
    
    return standardized


def merge_and_deduplicate(dataframes):
    """
    Merge datasets and remove duplicates.
    
    Args:
        dataframes: List of standardized dataframes
    
    Returns:
        pd.DataFrame: Merged and deduplicated dataframe
    """
    # Combine all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nMerged dataset: {len(merged_df)} samples before deduplication")
    
    # Remove duplicates based on text and label
    merged_df = merged_df.drop_duplicates(subset=['text', 'label'], keep='first')
    print(f"After removing duplicates: {len(merged_df)} samples")
    
    # Remove any rows with missing values
    initial_count = len(merged_df)
    merged_df = merged_df.dropna()
    if len(merged_df) < initial_count:
        print(f"Removed {initial_count - len(merged_df)} rows with missing values")
    
    return merged_df


def print_statistics(df):
    """
    Print dataset statistics.
    
    Args:
        df: Dataframe with 'text' and 'label' columns
    """
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print("\nClass Distribution:")
    
    class_dist = df['label'].value_counts().sort_index()
    for label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        class_name = "Non-Injection" if label == 0 else "Prompt Injection"
        print(f"  Label {label} ({class_name}): {count} samples ({percentage:.2f}%)")
    
    print(f"\nUnique samples: {len(df)}")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    
    # Define file paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "training" / "dataset.csv"
    excel_path = script_dir / "training" / "srinivasan_dataset.xlsx"  # Update if different path
    
    print("Hinglish Prompt Injection Dataset Merger")
    print("="*60)
    
    # Read datasets
    dataframes = read_datasets(str(csv_path), str(excel_path))
    
    # Standardize column names
    print("\nStandardizing column names...")
    standardized_dfs = standardize_columns(dataframes)
    
    # Merge and remove duplicates
    print("\nMerging datasets...")
    final_df = merge_and_deduplicate(standardized_dfs)
    
    # Print statistics
    print_statistics(final_df)
    
    # Save to output file
    output_path = script_dir / "training" / "master_train.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated dataset to: {output_path}")
    
    return final_df


if __name__ == "__main__":
    df = main()
