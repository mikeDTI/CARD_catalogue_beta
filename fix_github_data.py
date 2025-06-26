import pandas as pd
import re

def clean_text_for_tsv(text):
    """
    Clean text to be safe for TSV format by:
    1. Replacing newlines with spaces
    2. Replacing tabs with spaces
    3. Escaping quotes properly
    4. Truncating extremely long text
    """
    if pd.isna(text) or not str(text).strip():
        return ""
    
    text = str(text)
    
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Escape quotes if needed
    text = text.replace('"', '""')
    
    # Truncate if too long (keep first 1000 characters)
    if len(text) > 1000:
        text = text[:1000] + "... [truncated]"
    
    return text.strip()

def fix_github_data():
    """
    Fix the GitHub data file by cleaning problematic columns
    """
    print("Reading GitHub data file...")
    
    try:
        # Read the file with error handling
        df = pd.read_csv("github_repos_20250622_150029.tab", sep="\t", on_bad_lines='skip')
        print(f"Successfully read {len(df)} rows")
        
        # Clean the Code Summary column
        if 'Code Summary' in df.columns:
            print("Cleaning Code Summary column...")
            df['Code Summary'] = df['Code Summary'].apply(clean_text_for_tsv)
        
        # Clean the Data Types/Modalities column
        if 'Data Types/Modalities' in df.columns:
            print("Cleaning Data Types/Modalities column...")
            df['Data Types/Modalities'] = df['Data Types/Modalities'].apply(clean_text_for_tsv)
        
        # Clean the Tools/Packages column
        if 'Tools/Packages' in df.columns:
            print("Cleaning Tools/Packages column...")
            df['Tools/Packages'] = df['Tools/Packages'].apply(clean_text_for_tsv)
        
        # Save the cleaned data
        output_file = "github_repos_20250622_150029_cleaned.tab"
        df.to_csv(output_file, sep="\t", index=False, quoting=1)  # quoting=1 for QUOTE_ALL
        print(f"Saved cleaned data to {output_file}")
        
        # Display some statistics
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for any remaining issues
        for col in df.columns:
            if df[col].dtype == 'object':
                max_len = df[col].astype(str).str.len().max()
                print(f"{col}: max length = {max_len}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    fix_github_data() 