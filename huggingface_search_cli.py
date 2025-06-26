import pandas as pd
import requests
import time
import os
from typing import List, Dict
import json
import re
import sys
from datetime import datetime
from huggingface_hub import HfApi

def clean_text(text: str) -> str:
    """
    Remove newlines and extra whitespace from text
    """
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def get_model_details(model_id: str, api: HfApi) -> Dict:
    """
    Get detailed information about a model
    """
    try:
        # Get model info
        model_info = api.model_info(model_id)
        
        # Get model card content
        model_card = api.model_info(model_id, files_metadata=True)
        
        # Extract relevant information
        return {
            "model_id": model_id,
            "author": model_info.author,
            "last_modified": model_info.lastModified,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "tags": ", ".join(model_info.tags) if model_info.tags else "",
            "model_type": model_info.model_type if hasattr(model_info, 'model_type') else "",
            "library_name": model_info.library_name if hasattr(model_info, 'library_name') else "",
            "pipeline_tag": model_info.pipeline_tag if hasattr(model_info, 'pipeline_tag') else "",
            "description": clean_text(model_info.description) if model_info.description else ""
        }
    except Exception as e:
        print(f"Error getting details for model {model_id}: {str(e)}", file=sys.stderr)
        return None

def search_huggingface(study_name: str, abbreviation: str, diseases: str, data_modalities: str) -> List[Dict]:
    """
    Search Hugging Face Hub for models related to the study using study name and abbreviation
    """
    # Initialize Hugging Face API
    api = HfApi()
    
    # Build search terms - first study name, then abbreviation
    search_terms = []
    
    # Add study name for search with quotes for exact matching
    if study_name and study_name.strip():
        search_terms.append(f'"{study_name}"')
    
    # Add abbreviation for search with quotes for exact matching
    if abbreviation and abbreviation.strip():
        search_terms.append(f'"{abbreviation}"')
    
    # Log the search terms being used
    print(f"Search terms for {study_name} ({abbreviation}): {search_terms}", file=sys.stderr)
    
    results = []
    seen_model_ids = set()  # Track seen model IDs to avoid duplicates
    
    for term in search_terms:
        try:
            print(f"Searching for term: {term}", file=sys.stderr)
            
            # Search for models only
            models = api.list_models(
                search=term,
                limit=50,
                sort="downloads",
                direction=-1
            )
            
            for model in models:
                try:
                    # Skip if we've already seen this model
                    if model.modelId in seen_model_ids:
                        continue
                    
                    seen_model_ids.add(model.modelId)
                    
                    # Get basic model details only
                    model_details = {
                        "model_id": model.modelId,
                        "author": model.author if hasattr(model, 'author') else "",
                        "last_modified": model.lastModified if hasattr(model, 'lastModified') else "",
                        "downloads": model.downloads if hasattr(model, 'downloads') else 0,
                        "likes": model.likes if hasattr(model, 'likes') else 0,
                        "tags": ", ".join(model.tags) if hasattr(model, 'tags') and model.tags else "",
                        "model_type": model.model_type if hasattr(model, 'model_type') else "",
                        "library_name": model.library_name if hasattr(model, 'library_name') else "",
                        "pipeline_tag": model.pipeline_tag if hasattr(model, 'pipeline_tag') else "",
                        "description": clean_text(model.description) if hasattr(model, 'description') and model.description else "",
                        "Hugging Face Link": f"https://huggingface.co/{model.modelId}",
                        "Study Name": study_name,
                        "Abbreviation": abbreviation,
                        "Search Term": term
                    }
                    results.append(model_details)
                    
                    # Respect rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing model {model.modelId}: {str(e)}", file=sys.stderr)
                    continue
                    
        except Exception as e:
            print(f"Error searching for term {term}: {str(e)}", file=sys.stderr)
            continue
    
    return results

def main():
    # Read the dataset inventory
    try:
        studies_df = pd.read_csv("dataset-inventory-June_20_2025.tab", sep="\t")
    except Exception as e:
        print(f"Error reading dataset inventory: {str(e)}", file=sys.stderr)
        return
    
    # Initialize results list
    all_results = []
    
    # Process each study
    for _, row in studies_df.iterrows():
        study_name = row["Study Name"]
        abbreviation = row["Abbreviation"]
        diseases = row["Diseases Included"]
        data_modalities = row["Data Modalities"]
        
        print(f"Searching Hugging Face Hub for models related to {study_name} ({abbreviation})...", file=sys.stderr)
        results = search_huggingface(study_name, abbreviation, diseases, data_modalities)
        all_results.extend(results)
    
    # Create and save results dataframe
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"huggingface_models_{timestamp}.tab"
        results_df.to_csv(output_filename, sep="\t", index=False)
        print(f"Results saved to {output_filename}", file=sys.stderr)
    else:
        print("No results found", file=sys.stderr)

if __name__ == "__main__":
    main() 