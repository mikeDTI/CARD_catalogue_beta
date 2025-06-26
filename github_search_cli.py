import pandas as pd
import requests
import time
import os
from typing import List, Dict
import json
import re
import sys
from anthropic import Anthropic
from datetime import datetime

def clean_text(text: str) -> str:
    """
    Remove newlines and extra whitespace from text
    """
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def get_anthropic_client():
    """Initialize Anthropic client for version 0.55.0"""
    try:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            client = Anthropic(api_key=api_key)
            return client
        else:
            print("ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Failed to initialize Anthropic client: {e}", file=sys.stderr)
        return None

def get_ai_summary(code_content: str, api_key: str) -> Dict:
    """
    Use Claude to analyze code and extract summary, data types, and tools
    """
    if not code_content or not code_content.strip():
        return {
            "summary": "",
            "data_types": "",
            "tools": ""
        }
    
    try:
        # Initialize the Anthropic client using the working approach
        client = get_anthropic_client()
        if not client:
            return {
                "summary": "",
                "data_types": "",
                "tools": ""
            }
        
        # Create the analysis prompt
        prompt = f"""Analyze the following repository content and provide:

1. A concise one-paragraph summary of the code's purpose and functionality (no headers)
2. List of data types and data modalities mentioned or used (mirror the style from "Data Modalities" column)
3. List of packages, tools, and frameworks used

Repository content:
{code_content[:8000]}  # Limit content to avoid token limits

Provide the response in exactly 3 paragraphs separated by double newlines."""

        # Make the API call using the working approach
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the response content using the working approach
        content = response.content[0].text.replace('\n', ' ').replace('\r', ' ')
        
        # Parse the response into sections
        sections = content.split('\n\n')
        
        # Extract each section, handling potential parsing issues
        summary = clean_text(sections[0]) if len(sections) > 0 else ""
        data_types = clean_text(sections[1]) if len(sections) > 1 else ""
        tools = clean_text(sections[2]) if len(sections) > 2 else ""
        
        # Clean up any remaining newlines or formatting
        summary = summary.replace('\n', ' ').strip()
        data_types = data_types.replace('\n', ' ').strip()
        tools = tools.replace('\n', ' ').strip()
        
        return {
            "summary": summary,
            "data_types": data_types,
            "tools": tools
        }
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}", file=sys.stderr)
        return {
            "summary": "",
            "data_types": "",
            "tools": ""
        }

def search_github(study_name: str, abbreviation: str, diseases: str, github_token: str) -> List[Dict]:
    """
    Search GitHub for repositories related to the study using individual combinations
    """
    # Use only the exact disease keywords for searching
    disease_keywords = ["alzheimer", "parkinson", "dementia", "brain"]
    
    all_results = []
    seen_repos = set()  # Track seen repositories to avoid duplicates within study
    
    # Search combinations: study name + each disease keyword
    if study_name and study_name.strip():
        for disease_term in disease_keywords:
            query = f'"{study_name}" AND "{disease_term}"'
            print(f"  Searching: {query}", file=sys.stderr)
            results = search_github_with_query(query, study_name, abbreviation, diseases, github_token, seen_repos)
            all_results.extend(results)
    
    # Search combinations: abbreviation + each disease keyword
    if abbreviation and abbreviation.strip():
        for disease_term in disease_keywords:
            query = f'"{abbreviation}" AND "{disease_term}"'
            print(f"  Searching: {query}", file=sys.stderr)
            results = search_github_with_query(query, study_name, abbreviation, diseases, github_token, seen_repos)
            all_results.extend(results)
    
    return all_results

def search_github_with_query(query: str, study_name: str, abbreviation: str, diseases: str, github_token: str, seen_repos: set) -> List[Dict]:
    """
    Perform a single GitHub search with a specific query
    """
    # GitHub API endpoint
    url = 'https://api.github.com/search/repositories'
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100
    }
    
    try:
        # Add delay before making request to respect rate limits
        time.sleep(2)
        
        response = requests.get(url, headers=headers, params=params)
        
        # Check for rate limiting
        if response.status_code == 403:
            print(f"Rate limited for {study_name}. Waiting 60 seconds...", file=sys.stderr)
            time.sleep(60)  # Wait 60 seconds before retrying
            response = requests.get(url, headers=headers, params=params)
        
        response.raise_for_status()
        data = response.json()
        
        results = []
        for repo in data.get('items', []):
            try:
                # Skip if we've already seen this repository for this study
                repo_url = repo['html_url']
                if repo_url in seen_repos:
                    continue
                
                seen_repos.add(repo_url)
                
                # Get repository details
                owner = repo['owner']['login']
                languages = repo.get('language', '')
                
                # Get contributors with rate limiting
                contributors_url = repo['contributors_url']
                time.sleep(1)  # Add delay before contributors request
                contributors_response = requests.get(contributors_url, headers=headers)
                
                # Check for rate limiting on contributors
                if contributors_response.status_code == 403:
                    print(f"Rate limited during contributors fetch for {study_name}. Waiting 60 seconds...", file=sys.stderr)
                    time.sleep(60)
                    contributors_response = requests.get(contributors_url, headers=headers)
                
                contributors_response.raise_for_status()
                
                # Handle empty or invalid JSON responses
                contributors = []
                try:
                    contributors_data = contributors_response.json()
                    if contributors_data:
                        contributors = [c['login'] for c in contributors_data if isinstance(c, dict) and 'login' in c]
                except (ValueError, TypeError) as e:
                    print(f"Error parsing contributors for {repo['name']}: {str(e)}", file=sys.stderr)
                    contributors = []
                
                # Get README content with rate limiting
                readme_url = f"https://api.github.com/repos/{owner}/{repo['name']}/readme"
                time.sleep(1)  # Add delay before README request
                readme_response = requests.get(readme_url, headers=headers)
                
                # Check for rate limiting on README
                if readme_response.status_code == 403:
                    print(f"Rate limited during README fetch for {study_name}. Waiting 60 seconds...", file=sys.stderr)
                    time.sleep(60)
                    readme_response = requests.get(readme_url, headers=headers)
                
                readme_content = ""
                if readme_response.status_code == 200:
                    try:
                        readme_data = readme_response.json()
                        readme_content = readme_data.get('content', '') if readme_data else ""
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing README for {repo['name']}: {str(e)}", file=sys.stderr)
                        readme_content = ""
                
                # Get AI analysis
                ai_analysis = get_ai_summary(readme_content, os.getenv('ANTHROPIC_KEY'))
                
                # Skip if neither README nor code content is available
                if not readme_content and not ai_analysis["summary"]:
                    print(f"Skipping {repo_url} - no README or code content", file=sys.stderr)
                    continue
                
                results.append({
                    "Study Name": study_name,
                    "Abbreviation": abbreviation,
                    "Diseases Included": diseases,
                    "Repository Link": repo_url,
                    "Owner": owner,
                    "Contributors": "; ".join(contributors),
                    "Languages": languages,
                    "Code Summary": ai_analysis["summary"],
                    "Data Types/Modalities": ai_analysis["data_types"],
                    "Tools/Packages": ai_analysis["tools"],
                    "Search Query": query
                })
                
            except Exception as e:
                print(f"Error processing repository {repo['name']}: {str(e)}", file=sys.stderr)
                continue
                
        return results
        
    except Exception as e:
        print(f"Error searching GitHub with query '{query}': {str(e)}", file=sys.stderr)
        return []

def main():
    # Check for required environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    anthropic_key = os.getenv('ANTHROPIC_KEY')
    
    if not github_token or not anthropic_key:
        print("Error: Please set GITHUB_TOKEN and ANTHROPIC_KEY environment variables", file=sys.stderr)
        return
    
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
        
        print(f"Searching GitHub for repositories related to {study_name} ({abbreviation})...", file=sys.stderr)
        results = search_github(study_name, abbreviation, diseases, github_token)
        all_results.extend(results)
    
    # Remove redundant results for the same repository and study combinations
    print("Removing redundant results...", file=sys.stderr)
    seen_combinations = set()
    deduplicated_results = []
    
    for result in all_results:
        # Create a unique key for repository + study combination
        repo_link = result.get("Repository Link", "")
        study_name = result.get("Study Name", "")
        combination_key = f"{repo_link}_{study_name}"
        
        if combination_key not in seen_combinations:
            seen_combinations.add(combination_key)
            deduplicated_results.append(result)
        else:
            print(f"Removing duplicate: {repo_link} for {study_name}", file=sys.stderr)
    
    # Create and save results dataframe
    if deduplicated_results:
        results_df = pd.DataFrame(deduplicated_results)
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"github_repos_{timestamp}.tab"
        results_df.to_csv(output_filename, sep="\t", index=False)
        print(f"Results saved to {output_filename}", file=sys.stderr)
        print(f"Total results: {len(deduplicated_results)} (removed {len(all_results) - len(deduplicated_results)} duplicates)", file=sys.stderr)
    else:
        print("No results found", file=sys.stderr)

if __name__ == "__main__":
    main() 