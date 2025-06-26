import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import re
import sys

def clean_text(text):
    """
    Remove newlines and extra whitespace from text
    """
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def search_pubmed(study_name: str, abbreviation: str, diseases: str, data_modalities: str):
    """
    Search PubMed for articles related to the study
    """
    # Calculate date range (past 3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}"
    
    # Create search query with disease keywords and data modalities
    disease_keywords = ["alzheimer", "parkinson", "dementia", "brain", "neurodegenerative", "neurodegeneration", "tremor", "amyotrophic"]
    disease_terms = []
    if pd.notna(diseases) and isinstance(diseases, str):
        disease_terms = [d.lower() for d in diseases.split(";") if any(kw in d.lower() for kw in disease_keywords)]
    
    if not disease_terms:
        disease_terms = disease_keywords
    
    # Extract data modalities
    modalities = []
    if pd.notna(data_modalities) and isinstance(data_modalities, str):
        modalities = [m.strip() for m in data_modalities.strip('[]').split(';')]
    
    # Build query terms
    query_terms = [
        f'("{study_name}"[All Fields] OR "{abbreviation}"[All Fields])',
        f'({" OR ".join([f"{term}[All Fields]" for term in disease_terms])})',
        f'({date_range}[Date - Publication])'
    ]
    
    # Add data modalities to query if available
    if modalities:
        modality_terms = [f'"{modality}"[All Fields]' for modality in modalities]
        query_terms.append(f'({" OR ".join(modality_terms)})')
    
    query = " AND ".join(query_terms)
    
    # Search PubMed
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    url = f'{base_url}?db=pubmed&term={query}&retmax=100&retmode=json'
    
    try:
        # Add delay before making request to respect rate limits
        time.sleep(1)
        
        response = requests.get(url)
        
        # Check for rate limiting
        if response.status_code == 429:
            print(f"Rate limited for {study_name}. Waiting 60 seconds...", file=sys.stderr)
            time.sleep(60)  # Wait 60 seconds before retrying
            response = requests.get(url)
        
        response.raise_for_status()
        data = response.json()
        pubmed_ids = data['esearchresult']['idlist']
        
        results = []
        for pubmed_id in pubmed_ids:
            try:
                # Get article details
                fetch_url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml'
                
                # Add delay before each fetch request
                time.sleep(1)
                
                fetch_response = requests.get(fetch_url)
                
                # Check for rate limiting on fetch requests
                if fetch_response.status_code == 429:
                    print(f"Rate limited during fetch for {study_name}. Waiting 60 seconds...", file=sys.stderr)
                    time.sleep(60)
                    fetch_response = requests.get(fetch_url)
                
                fetch_response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(fetch_response.text)
                
                # Extract article information
                article = root.find('.//PubmedArticle')
                if article is None:
                    continue
                
                # Get title
                title = article.find('.//ArticleTitle')
                title = clean_text(title.text if title is not None else "")
                
                # Get abstract
                abstract = ""
                abstract_elements = article.findall('.//AbstractText')
                if abstract_elements:
                    abstract = " ".join(clean_text(elem.text) for elem in abstract_elements if elem.text)
                
                # Get authors and affiliations
                authors = []
                affiliations = []
                author_list = article.find('.//AuthorList')
                if author_list is not None:
                    for author in author_list.findall('.//Author'):
                        last_name = author.find('.//LastName')
                        fore_name = author.find('.//ForeName')
                        if last_name is not None and fore_name is not None:
                            authors.append(f"{last_name.text} {fore_name.text}")
                        
                        aff = author.find('.//Affiliation')
                        if aff is not None and aff.text:
                            affiliations.append(clean_text(aff.text))
                
                # Get keywords
                keywords = []
                keyword_list = article.find('.//KeywordList')
                if keyword_list is not None:
                    keywords = [k.text for k in keyword_list.findall('.//Keyword') if k.text]
                
                # Get PMC ID if available
                pmc_id = None
                article_ids = article.findall('.//ArticleId')
                for article_id in article_ids:
                    if article_id.get('IdType') == 'pmc':
                        pmc_id = article_id.text
                        break
                
                # Create PMC link if available
                pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/" if pmc_id else ""
                
                results.append({
                    "Study Name": study_name,
                    "Abbreviation": abbreviation,
                    "Diseases Included": diseases,
                    "Data Modalities": data_modalities,
                    "PubMed Central Link": pmc_link,
                    "Authors": "; ".join(authors),
                    "Affiliations": "; ".join(affiliations),
                    "Title": title,
                    "Abstract": abstract,
                    "Keywords": "; ".join(keywords)
                })
                
            except Exception as e:
                print(f"Error processing article {pubmed_id}: {str(e)}", file=sys.stderr)
                continue
                
        return results
        
    except Exception as e:
        print(f"Error searching for {study_name}: {str(e)}", file=sys.stderr)
        return []

def main():
    # Read the dataset inventory
    try:
        studies_df = pd.read_csv("studies_to_add_revision-June_24_2025.tab", sep="\t")
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
        
        print(f"Searching for publications related to {study_name} ({abbreviation})...", file=sys.stderr)
        results = search_pubmed(study_name, abbreviation, diseases, data_modalities)
        all_results.extend(results)
    
    # Create and save results dataframe
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pubmed_central_{timestamp}.tab"
        results_df.to_csv(output_filename, sep="\t", index=False)
        print(f"Results saved to {output_filename}", file=sys.stderr)
    else:
        print("No results found", file=sys.stderr)

if __name__ == "__main__":
    main() 