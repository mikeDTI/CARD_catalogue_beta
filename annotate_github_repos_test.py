import pandas as pd
import requests
import json
import time
import os
from urllib.parse import urlparse
import base64

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

def get_github_repo_info(repo_url):
    """
    Extract repository information from GitHub URL
    """
    try:
        # Parse the GitHub URL
        parsed = urlparse(repo_url)
        if 'github.com' not in parsed.netloc:
            return None, None
        
        # Extract owner and repo name
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]
            return owner, repo
        return None, None
    except:
        return None, None

def get_github_content(owner, repo, path="", token=None):
    """
    Get content from GitHub repository using GitHub API
    """
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'CARD-Catalogue-Annotator'
    }
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Get repository README
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(readme_url, headers=headers, timeout=10)
        if response.status_code == 200:
            content = response.json()
            if content.get('content'):
                # Decode base64 content
                readme_content = base64.b64decode(content['content']).decode('utf-8')
                return readme_content
    except Exception as e:
        print(f"Error getting README for {owner}/{repo}: {e}")
    
    # If README not found, try to get repository description
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(repo_url, headers=headers, timeout=10)
        if response.status_code == 200:
            repo_info = response.json()
            description = repo_info.get('description', '')
            topics = repo_info.get('topics', [])
            language = repo_info.get('language', '')
            
            content = f"Repository: {repo}\n"
            if description:
                content += f"Description: {description}\n"
            if topics:
                content += f"Topics: {', '.join(topics)}\n"
            if language:
                content += f"Primary Language: {language}\n"
            
            return content
    except Exception as e:
        print(f"Error getting repo info for {owner}/{repo}: {e}")
    
    return ""

def analyze_repository_with_anthropic(repo_url, repo_content):
    """
    Use Anthropic to analyze repository content and extract information
    """
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not found in environment variables")
        return None, None, None, None
    
    prompt = f"""
You are analyzing a GitHub repository for a biomedical research catalog. Please analyze the following repository information and provide four specific outputs:

Repository URL: {repo_url}

Repository Content:
{repo_content}

Please provide four separate analyses:

1. BIOMEDICAL RELEVANCE: First, determine if this repository is related to biomedical research, healthcare, neuroscience, or medical applications. Answer with "YES" or "NO" and provide a brief explanation (1-2 sentences).

2. CODE SUMMARY: Write a brief summary of the repository's purpose and functionality (less than 300 words). Focus on what the code does, its main features, and its relevance to biomedical research (if applicable).

3. DATA TYPES: Identify and describe the types of data mentioned or used in this repository. Include data modalities, file formats, and data sources if mentioned.

4. TOOLING: List and describe the analytics tools, packages, frameworks, and technologies used in this repository.

Format your response exactly as follows:

BIOMEDICAL RELEVANCE:
[YES/NO] - [Brief explanation]

CODE SUMMARY:
[Your summary here]

DATA TYPES:
[Your data types analysis here]

TOOLING:
[Your tooling analysis here]

If any section cannot be determined from the available information, write "Not specified" for that section.
"""

    try:
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 2000,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            
            # Parse the response
            biomedical_relevance = ""
            code_summary = ""
            data_types = ""
            tooling = ""
            
            # Extract biomedical relevance
            if "BIOMEDICAL RELEVANCE:" in content:
                relevance_start = content.find("BIOMEDICAL RELEVANCE:") + len("BIOMEDICAL RELEVANCE:")
                relevance_end = content.find("CODE SUMMARY:")
                if relevance_end == -1:
                    relevance_end = content.find("DATA TYPES:")
                if relevance_end == -1:
                    relevance_end = content.find("TOOLING:")
                if relevance_end == -1:
                    relevance_end = len(content)
                
                biomedical_relevance = content[relevance_start:relevance_end].strip()
            
            # Extract code summary
            if "CODE SUMMARY:" in content:
                code_summary_start = content.find("CODE SUMMARY:") + len("CODE SUMMARY:")
                code_summary_end = content.find("DATA TYPES:")
                if code_summary_end == -1:
                    code_summary_end = content.find("TOOLING:")
                if code_summary_end == -1:
                    code_summary_end = len(content)
                
                code_summary = content[code_summary_start:code_summary_end].strip()
            
            # Extract data types
            if "DATA TYPES:" in content:
                data_types_start = content.find("DATA TYPES:") + len("DATA TYPES:")
                data_types_end = content.find("TOOLING:")
                if data_types_end == -1:
                    data_types_end = len(content)
                
                data_types = content[data_types_start:data_types_end].strip()
            
            # Extract tooling
            if "TOOLING:" in content:
                tooling_start = content.find("TOOLING:") + len("TOOLING:")
                tooling = content[tooling_start:].strip()
            
            return biomedical_relevance, code_summary, data_types, tooling
        else:
            print(f"API request failed with status {response.status_code}: {response.text}")
            return None, None, None, None
            
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return None, None, None, None

def clean_text_for_tsv(text):
    """
    Clean text to be safe for TSV format
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    # Replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Replace multiple spaces with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    # Truncate if too long
    if len(text) > 1000:
        text = text[:1000] + "... [truncated]"
    
    return text.strip()

def main():
    """
    Main function to process the GitHub repositories (test version - first 5 only)
    """
    print("Starting GitHub repository annotation (TEST MODE - first 5 repositories)...")
    
    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it using: export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Read the input file
    try:
        df = pd.read_csv("gits_to_reannotate.tsv", sep="\t")
        print(f"Loaded {len(df)} repositories to process")
        
        # Limit to first 5 for testing
        df = df.head(5)
        print(f"Processing first {len(df)} repositories for testing")
        
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Add new columns
    df['Biomedical Relevance'] = ""
    df['Code Summary'] = ""
    df['Data Types'] = ""
    df['Tooling'] = ""
    
    # Process each repository
    for index, row in df.iterrows():
        repo_url = row['Repository Link']
        print(f"\nProcessing {index + 1}/{len(df)}: {repo_url}")
        
        # Extract repository info
        owner, repo = get_github_repo_info(repo_url)
        if not owner or not repo:
            print(f"  Could not parse repository URL: {repo_url}")
            continue
        
        # Get repository content
        repo_content = get_github_content(owner, repo)
        if not repo_content:
            print(f"  No content found for {owner}/{repo}")
            continue
        
        print(f"  Retrieved content for {owner}/{repo} ({len(repo_content)} characters)")
        
        # Analyze with Anthropic
        biomedical_relevance, code_summary, data_types, tooling = analyze_repository_with_anthropic(repo_url, repo_content)
        
        if biomedical_relevance is not None:
            # Clean and store results
            df.at[index, 'Biomedical Relevance'] = clean_text_for_tsv(biomedical_relevance)
            df.at[index, 'Code Summary'] = clean_text_for_tsv(code_summary)
            df.at[index, 'Data Types'] = clean_text_for_tsv(data_types)
            df.at[index, 'Tooling'] = clean_text_for_tsv(tooling)
            print(f"  Successfully analyzed {owner}/{repo}")
            print(f"    Biomedical Relevance: {biomedical_relevance[:100]}...")
            print(f"    Code Summary: {len(code_summary)} characters")
            print(f"    Data Types: {len(data_types)} characters")
            print(f"    Tooling: {len(tooling)} characters")
        else:
            print(f"  Failed to analyze {owner}/{repo}")
        
        # Rate limiting - wait between requests
        time.sleep(3)
    
    # Save results
    output_file = "gits_to_reannotate_test_annotated.tsv"
    df.to_csv(output_file, sep="\t", index=False)
    print(f"\nTest annotation complete! Results saved to {output_file}")
    
    # Print summary
    processed_count = df[df['Code Summary'] != ''].shape[0]
    print(f"Successfully processed {processed_count}/{len(df)} repositories")
    
    # Show sample results and flag non-biomedical repos
    print("\nSample results:")
    non_biomedical_count = 0
    for index, row in df.iterrows():
        if row['Code Summary']:
            relevance = row['Biomedical Relevance']
            is_biomedical = "YES" in relevance.upper() if relevance else False
            
            if not is_biomedical:
                non_biomedical_count += 1
                print(f"\n⚠️  NON-BIOMEDICAL REPOSITORY: {row['Repository Link']}")
                print(f"  Relevance: {relevance}")
            else:
                print(f"\n✅ BIOMEDICAL REPOSITORY: {row['Repository Link']}")
                print(f"  Relevance: {relevance}")
            
            print(f"  Summary: {row['Code Summary'][:100]}...")
            print(f"  Data Types: {row['Data Types'][:100]}...")
            print(f"  Tooling: {row['Tooling'][:100]}...")
    
    print(f"\nSummary: {non_biomedical_count} out of {processed_count} repositories appear to be non-biomedical")

if __name__ == "__main__":
    main() 