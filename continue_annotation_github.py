import pandas as pd
import requests
import json
import time
import os
from urllib.parse import urlparse
import base64
from datetime import datetime
import glob
from bs4 import BeautifulSoup

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
    Get content from GitHub repository using GitHub API and web scraping
    """
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'CARD-Catalogue-Annotator'
    }
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    content_parts = []
    
    # Get repository metadata first
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(repo_url, headers=headers, timeout=10)
        if response.status_code == 200:
            repo_info = response.json()
            description = repo_info.get('description', '')
            topics = repo_info.get('topics', [])
            language = repo_info.get('language', '')
            homepage = repo_info.get('homepage', '')
            stargazers_count = repo_info.get('stargazers_count', 0)
            forks_count = repo_info.get('forks_count', 0)
            created_at = repo_info.get('created_at', '')
            updated_at = repo_info.get('updated_at', '')
            
            content_parts.append(f"Repository: {owner}/{repo}")
            if description:
                content_parts.append(f"Description: {description}")
            if topics:
                content_parts.append(f"Topics: {', '.join(topics)}")
            if language:
                content_parts.append(f"Primary Language: {language}")
            if homepage:
                content_parts.append(f"Homepage: {homepage}")
            if stargazers_count > 0:
                content_parts.append(f"Stars: {stargazers_count}")
            if forks_count > 0:
                content_parts.append(f"Forks: {forks_count}")
            if created_at:
                content_parts.append(f"Created: {created_at}")
            if updated_at:
                content_parts.append(f"Last Updated: {updated_at}")
    except Exception as e:
        print(f"Error getting repo info for {owner}/{repo}: {e}")
    
    # Scrape the repository landing page
    repo_web_url = f"https://github.com/{owner}/{repo}"
    try:
        web_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(repo_web_url, headers=web_headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract README content from the main page
            readme_div = soup.find('div', {'data-testid': 'readme'})
            if readme_div:
                readme_content = readme_div.get_text(separator='\n', strip=True)
                if readme_content and len(readme_content.strip()) > 50:
                    print(f"    Found README content from web page ({len(readme_content)} characters)")
                    content_parts.append(f"\nREADME (from web):\n{readme_content}")
            
            # Extract repository description from the page
            description_elem = soup.find('div', {'class': 'f4 mb-3'})
            if description_elem:
                page_description = description_elem.get_text(strip=True)
                if page_description and page_description not in content_parts:
                    content_parts.append(f"Page Description: {page_description}")
            
            # Extract topics/tags from the page
            topics_container = soup.find('div', {'class': 'topics-row-container'})
            if topics_container:
                topic_links = topics_container.find_all('a', {'class': 'topic-tag'})
                if topic_links:
                    page_topics = [link.get_text(strip=True) for link in topic_links]
                    if page_topics and page_topics != topics:
                        content_parts.append(f"Page Topics: {', '.join(page_topics)}")
            
            # Extract repository stats
            stats_container = soup.find('div', {'class': 'd-flex flex-wrap gap-2'})
            if stats_container:
                stats_text = stats_container.get_text(strip=True)
                if stats_text:
                    content_parts.append(f"Repository Stats: {stats_text}")
                    
    except Exception as e:
        print(f"Error scraping web page for {owner}/{repo}: {e}")
    
    # Get repository README via API as backup
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(readme_url, headers=headers, timeout=10)
        if response.status_code == 200:
            content = response.json()
            if content.get('content'):
                # Decode base64 content
                readme_content = base64.b64decode(content['content']).decode('utf-8')
                if readme_content and readme_content.strip():
                    print(f"    Found README content via API ({len(readme_content)} characters)")
                    content_parts.append(f"\nREADME (via API):\n{readme_content}")
    except Exception as e:
        print(f"Error getting README for {owner}/{repo}: {e}")
    
    # Get repository files and structure
    try:
        contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = requests.get(contents_url, headers=headers, timeout=10)
        if response.status_code == 200:
            contents = response.json()
            if isinstance(contents, list) and len(contents) > 0:
                # Get first few files to understand the repository
                file_info = []
                directories = []
                
                for item in contents[:10]:  # Increased to 10 files
                    if item.get('type') == 'file':
                        file_name = item.get('name', '')
                        file_size = item.get('size', 0)
                        # Only include files that might be informative
                        if any(ext in file_name.lower() for ext in ['.md', '.txt', '.py', '.r', '.js', '.html', '.ipynb', '.yml', '.yaml', '.json', '.xml', '.csv', '.tsv']):
                            file_info.append(f"File: {file_name} ({file_size} bytes)")
                    elif item.get('type') == 'dir':
                        dir_name = item.get('name', '')
                        directories.append(f"Directory: {dir_name}")
                
                if file_info or directories:
                    structure_parts = []
                    if directories:
                        structure_parts.append("Directories:")
                        structure_parts.extend(directories[:5])  # Limit directories
                    if file_info:
                        structure_parts.append("Key Files:")
                        structure_parts.extend(file_info)
                    
                    content_parts.append(f"\nRepository Structure:\n" + "\n".join(structure_parts))
                    print(f"    Found repository structure with {len(file_info)} files and {len(directories)} directories")
    except Exception as e:
        print(f"Error getting repository contents for {owner}/{repo}: {e}")
    
    # Try to get additional documentation files
    doc_files = ['README.md', 'README.txt', 'docs/README.md', 'documentation.md', 'ABOUT.md']
    for doc_file in doc_files:
        try:
            doc_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{doc_file}"
            response = requests.get(doc_url, headers=headers, timeout=10)
            if response.status_code == 200:
                content = response.json()
                if content.get('content'):
                    doc_content = base64.b64decode(content['content']).decode('utf-8')
                    if doc_content and doc_content.strip():
                        print(f"    Found additional documentation: {doc_file} ({len(doc_content)} characters)")
                        content_parts.append(f"\n{doc_file}:\n{doc_content[:1000]}...")  # Limit length
                        break
        except Exception:
            continue
    
    # Combine all content
    final_content = "\n".join(content_parts)
    
    if final_content and final_content.strip():
        print(f"    Total content collected: {len(final_content)} characters")
        return final_content
    else:
        print(f"    No content found for {owner}/{repo}")
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

def is_already_processed(row):
    """
    Check if a repository has already been processed (has biomedical relevance and code summary)
    """
    try:
        biomedical_relevance = str(row.get('Biomedical Relevance', '')).strip()
        code_summary = str(row.get('Code Summary', '')).strip()
        
        # Check if both fields have meaningful content (not empty, not NaN, not "nan")
        has_relevance = (biomedical_relevance and 
                        biomedical_relevance != "" and 
                        biomedical_relevance.lower() != "nan")
        has_summary = (code_summary and 
                      code_summary != "" and 
                      code_summary.lower() != "nan")
        
        return has_relevance and has_summary
    except Exception as e:
        print(f"Error checking if already processed: {e}")
        return False

def main():
    """
    Main function to continue processing from existing annotated file
    """
    print("Starting continued GitHub repository annotation...")
    
    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it using: export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Find the most recent completed file
    completed_files = glob.glob("gits_to_reannotate_completed_*.tsv")
    if not completed_files:
        print("No completed files found! Looking for any annotated files...")
        annotated_files = glob.glob("gits_to_reannotate_*_annotated.tsv")
        if not annotated_files:
            print("No annotated files found! Please run the full annotation script first.")
            return
        input_file = max(annotated_files, key=os.path.getctime)
        print(f"Using most recent annotated file: {input_file}")
    else:
        input_file = max(completed_files, key=os.path.getctime)
        print(f"Using most recent completed file: {input_file}")
    
    # Read the existing annotated file
    try:
        df = pd.read_csv(input_file, sep="\t")
        print(f"Loaded {len(df)} repositories from {input_file}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Please run the full annotation script first or ensure the file exists.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Ensure required columns exist
    required_columns = ['Biomedical Relevance', 'Code Summary', 'Data Types', 'Tooling']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
            print(f"Added missing column: {col}")
    
    # Identify repositories that need processing
    unprocessed_indices = []
    processed_count = 0
    
    print("Checking which repositories need processing...")
    for index, row in df.iterrows():
        try:
            if is_already_processed(row):
                processed_count += 1
            else:
                unprocessed_indices.append(index)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            unprocessed_indices.append(index)
    
    print(f"Already processed: {processed_count} repositories")
    print(f"Remaining to process: {len(unprocessed_indices)} repositories")
    
    if len(unprocessed_indices) == 0:
        print("All repositories have already been processed!")
        return
    
    # Track statistics
    total_processed_this_run = 0
    biomedical_count = 0
    non_biomedical_count = 0
    
    # Process unprocessed repositories
    for i, index in enumerate(unprocessed_indices):
        try:
            row = df.iloc[index]
            repo_url = row['Repository Link']
            print(f"\nProcessing {i + 1}/{len(unprocessed_indices)}: {repo_url}")
            
            # Extract repository info
            owner, repo = get_github_repo_info(repo_url)
            if not owner or not repo:
                print(f"  Could not parse repository URL: {repo_url}")
                continue
            
            # Get repository content
            repo_content = get_github_content(owner, repo)
            if not repo_content or len(repo_content.strip()) < 5:  # Reduced from 10 to 5 characters
                print(f"  No meaningful content found for {owner}/{repo}")
                continue
            
            # Additional quality check - ensure we have at least some descriptive content
            content_quality = False
            if any(keyword in repo_content.lower() for keyword in ['description:', 'topics:', 'readme:', 'repository:', 'language:']):
                content_quality = True
            elif len(repo_content.strip()) > 20:  # If content is substantial, accept it
                content_quality = True
            
            if not content_quality:
                print(f"  Content quality too low for {owner}/{repo}")
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
                
                # Track statistics
                total_processed_this_run += 1
                is_biomedical = "YES" in biomedical_relevance.upper() if biomedical_relevance else False
                
                if is_biomedical:
                    biomedical_count += 1
                    print(f"  ✅ BIOMEDICAL: {owner}/{repo}")
                else:
                    non_biomedical_count += 1
                    print(f"  ⚠️  NON-BIOMEDICAL: {owner}/{repo}")
                
                print(f"    Relevance: {biomedical_relevance[:100]}...")
            else:
                print(f"  Failed to analyze {owner}/{repo}")
            
            # Rate limiting - wait between requests
            time.sleep(2)
            
            # Save progress every 10 repositories
            if (i + 1) % 10 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                progress_file = f"gits_to_reannotate_continued_{timestamp}.tsv"
                df.to_csv(progress_file, sep="\t", index=False)
                print(f"  Progress saved to {progress_file}")
                print(f"  Current stats: {biomedical_count} biomedical, {non_biomedical_count} non-biomedical")
                
        except Exception as e:
            print(f"  Error processing repository at index {index}: {e}")
            continue
    
    # Save final results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gits_to_reannotate_completed_{timestamp}.tsv"
    df.to_csv(output_file, sep="\t", index=False)
    print(f"\nAnnotation complete! Results saved to {output_file}")
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total repositories in file: {len(df)}")
    print(f"Previously processed: {processed_count}")
    print(f"Processed in this run: {total_processed_this_run}")
    print(f"Biomedical repositories (this run): {biomedical_count}")
    print(f"Non-biomedical repositories (this run): {non_biomedical_count}")
    
    # Calculate overall statistics
    try:
        # Convert to string and handle NaN values properly
        biomedical_col = df['Biomedical Relevance'].astype(str).replace('nan', '')
        code_summary_col = df['Code Summary'].astype(str).replace('nan', '')
        
        total_biomedical = biomedical_col.str.contains('YES', case=False, na=False).sum()
        total_non_biomedical = biomedical_col.str.contains('NO', case=False, na=False).sum()
        total_processed = (code_summary_col != '').sum()
        
        print(f"\n=== OVERALL STATISTICS ===")
        print(f"Total processed: {total_processed}")
        print(f"Total biomedical: {total_biomedical}")
        print(f"Total non-biomedical: {total_non_biomedical}")
        print(f"Biomedical percentage: {(total_biomedical/total_processed*100):.1f}%" if total_processed > 0 else "N/A")
        
        # List non-biomedical repositories
        if total_non_biomedical > 0:
            print(f"\n=== NON-BIOMEDICAL REPOSITORIES ===")
            for index, row in df.iterrows():
                relevance = str(row.get('Biomedical Relevance', '')).replace('nan', '')
                if relevance and "NO" in relevance.upper():
                    repo_link = row.get('Repository Link', 'Unknown')
                    print(f"  {repo_link}: {relevance[:100]}...")
    except Exception as e:
        print(f"Error calculating final statistics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 