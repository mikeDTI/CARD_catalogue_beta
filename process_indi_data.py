import pandas as pd
import os
from datetime import datetime
import anthropic
import time
import sys

# Note: You need to set your Anthropic API key as an environment variable:
# export ANTHROPIC_API_KEY="your-api-key-here"
# Or on Windows: set ANTHROPIC_API_KEY=your-api-key-here

def clean_column_names(df):
    """Remove (D365) suffixes from column headers"""
    df.columns = df.columns.str.replace(' \(D365\)', '', regex=True)
    return df

def get_gene_summary(gene_name, client, gene_cache=None):
    """Get gene summary using Claude 4"""
    if pd.isna(gene_name) or gene_name == "":
        return "N/A"
    
    # Check cache first
    if gene_cache is not None and gene_name in gene_cache:
        return gene_cache[gene_name]
    
    prompt = f"""Please provide a concise summary (no more than one paragraph) of the gene {gene_name} and its implications in neurodegenerative diseases. Use PubMed Central (PMC) as your primary source for scientific literature. Include relevant citations from PMC articles. Focus on the most important findings and clinical relevance. Provide the summary as a single paragraph without line breaks."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=750,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        summary = response.content[0].text.replace('\n', ' ').replace('\r', ' ')
        # Store in cache
        if gene_cache is not None:
            gene_cache[gene_name] = summary
        return summary
    except Exception as e:
        print(f"Error getting gene summary for {gene_name}: {str(e)}", file=sys.stderr)
        return "N/A"

def get_variant_summary(dbSNP_id, client, variant_cache=None):
    """Get variant summary using Claude 4"""
    if pd.isna(dbSNP_id) or dbSNP_id == "":
        return "N/A"
    
    # Check cache first
    if variant_cache is not None and dbSNP_id in variant_cache:
        return variant_cache[dbSNP_id]
    
    prompt = f"""Please summarize whatever you find about {dbSNP_id} in a single paragraph. Summarize the findings like a scientist would"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=750,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        summary = response.content[0].text.replace('\n', ' ').replace('\r', ' ')
        # Store in cache
        if variant_cache is not None:
            variant_cache[dbSNP_id] = summary
        return summary
    except Exception as e:
        print(f"Error getting variant summary for {dbSNP_id}: {str(e)}", file=sys.stderr)
        return "N/A"

def main():
    # File paths
    excel_file = "./iNDI_metadata/IPSC Data 6-18-2025 13-15.xlsx"
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"Error: File {excel_file} not found", file=sys.stderr)
        return
    
    try:
        # Read BackgroundInfo tab, skipping the second row
        print("Reading BackgroundInfo tab...", file=sys.stderr)
        background_df = pd.read_excel(
            excel_file, 
            sheet_name="BackgroundInfo", 
            skiprows=[1]  # Skip second row (index 1)
        )
        
        # Read DiseaseList tab
        print("Reading DiseaseList tab...", file=sys.stderr)
        disease_df = pd.read_excel(excel_file, sheet_name="DiseaseList")
        
        # Extract required columns from BackgroundInfo
        required_columns = [
            "Product Code", "Parental Line (D365)", "Gene (D365)", 
            "Gene Variant (D365)", "Genotype (D365)", "dbSNP", 
            "Condition", "Other Names", "Genome Assembly", 
            "Protospacer Sequence", "Genomic Coordinate", "Genomic Sequence"
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in background_df.columns]
        if missing_columns:
            print(f"Error: Missing columns in BackgroundInfo tab: {missing_columns}", file=sys.stderr)
            print(f"Available columns: {list(background_df.columns)}", file=sys.stderr)
            return
        
        # Create new dataframe with required columns
        result_df = background_df[required_columns].copy()
        
        # Clean column names (remove D365 suffixes)
        result_df = clean_column_names(result_df)
        
        # Replace Condition with geneticCondition from DiseaseList tab
        print("Matching conditions with DiseaseList...", file=sys.stderr)
        if 'geneticConditionId' in disease_df.columns and 'geneticCondition' in disease_df.columns:
            # Create a mapping dictionary
            condition_mapping = dict(zip(disease_df['geneticConditionId'], disease_df['geneticCondition']))
            
            # Replace Condition values
            result_df['Condition'] = result_df['Condition'].map(condition_mapping).fillna(result_df['Condition'])
        else:
            print("Warning: geneticConditionId or geneticCondition columns not found in DiseaseList tab", file=sys.stderr)
        
        # Add Procurement link column
        result_df['Procurement link'] = "https://www.jax.org/jax-mice-and-services/ipsc/cells-collection"
        
        # Initialize Claude client and caches
        print("Initializing Claude client...", file=sys.stderr)
        client = anthropic.Anthropic()
        gene_cache = {}
        variant_cache = {}
        
        # Add gene summaries
        print("Generating gene summaries...", file=sys.stderr)
        for idx, row in result_df.iterrows():
            gene_name = row['Gene']
            product_code = row['Product Code']
            print(f"Processing gene summary for Product Code: {product_code}, Gene: {gene_name}", file=sys.stderr)
            result_df.at[idx, 'About this gene'] = get_gene_summary(gene_name, client, gene_cache)
        
        # Add small delay to respect rate limits
        time.sleep(2)
        
        # Add variant summaries
        print("Generating variant summaries...", file=sys.stderr)
        for idx, row in result_df.iterrows():
            dbSNP_id = row['dbSNP']
            product_code = row['Product Code']
            print(f"Processing variant summary for Product Code: {product_code}, dbSNP: {dbSNP_id}", file=sys.stderr)
            result_df.at[idx, 'About this variant'] = get_variant_summary(dbSNP_id, client, variant_cache)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"iNDI_inventory_{timestamp}.tab"
        
        # Save to tab-delimited file
        print(f"Saving results to {output_filename}...", file=sys.stderr)
        result_df.to_csv(output_filename, sep='\t', index=False)
        
        print(f"Successfully processed {len(result_df)} records", file=sys.stderr)
        print(f"Results saved to: {output_filename}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        return

if __name__ == "__main__":
    main() 