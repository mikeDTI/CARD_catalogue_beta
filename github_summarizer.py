#!/usr/bin/env python3
import requests
import sys
import os
import re

# Import anthropic for version 0.55.0
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
    print(f"Anthropic version: {Anthropic.__version__}", file=sys.stderr)
except Exception as e:
    print(f"Anthropic import failed: {e}", file=sys.stderr)
    ANTHROPIC_AVAILABLE = False

def clean_text(text: str) -> str:
    """Remove newlines and extra whitespace from text"""
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def get_anthropic_client():
    """Initialize Anthropic client for version 0.55.0"""
    if not ANTHROPIC_AVAILABLE:
        return None
    
    try:
        # For version 0.55.0, use the standard initialization
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            client = Anthropic(api_key=api_key)
            print("Successfully initialized Anthropic client (v0.55.0)", file=sys.stderr)
            return client
        else:
            print("ANTHROPIC_API_KEY not set", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Anthropic initialization failed: {e}", file=sys.stderr)
        return None

def get_github_content(github_url: str) -> str:
    """Extract content from GitHub repository"""
    try:
        # Parse GitHub URL to get owner and repo name
        # Handle different GitHub URL formats
        if 'github.com' in github_url:
            parts = github_url.split('github.com/')[-1].split('/')
            if len(parts) >= 2:
                owner = parts[0]
                repo = parts[1]
            else:
                print("Invalid GitHub URL format", file=sys.stderr)
                return ""
        else:
            print("Not a GitHub URL", file=sys.stderr)
            return ""
        
        # Get repository information
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            print(f"Error accessing repository: {response.status_code}", file=sys.stderr)
            return ""
        
        repo_data = response.json()
        
        # Get README content
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_response = requests.get(readme_url)
        
        content_parts = []
        
        # Add repository description
        if repo_data.get('description'):
            content_parts.append(f"Repository Description: {repo_data['description']}")
        
        # Add repository topics
        if repo_data.get('topics'):
            content_parts.append(f"Topics: {', '.join(repo_data['topics'])}")
        
        # Add README content
        if readme_response.status_code == 200:
            try:
                readme_data = readme_response.json()
                readme_content = readme_data.get('content', '')
                if readme_content:
                    content_parts.append(f"README Content: {readme_content}")
            except Exception as e:
                print(f"Error parsing README: {e}", file=sys.stderr)
        
        # Add language information
        if repo_data.get('language'):
            content_parts.append(f"Primary Language: {repo_data['language']}")
        
        # Add star count
        if repo_data.get('stargazers_count'):
            content_parts.append(f"Stars: {repo_data['stargazers_count']}")
        
        return "\n\n".join(content_parts)
        
    except Exception as e:
        print(f"Error getting GitHub content: {e}", file=sys.stderr)
        return ""

def summarize_github_repo(github_url: str) -> str:
    """Summarize a GitHub repository using AI or keyword analysis"""
    try:
        # Get content from GitHub
        content = get_github_content(github_url)
        
        if not content:
            return "Unable to retrieve repository content"
        
        # Try AI summary first
        client = get_anthropic_client()
        if client:
            try:
                # Create summary prompt
                prompt = f"""Analyze the following GitHub repository and provide a comprehensive summary including:

1. What the repository does and its main purpose
2. Key features and functionality
3. Technologies and tools used
4. Target audience or use cases
5. Notable aspects or highlights

Repository information:
{content[:8000]}

Provide a clear, well-structured summary in 2-3 paragraphs."""

                # Get AI summary with multiple fallback approaches
                models_to_try = [
                    "claude-sonnet-4-20250514",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
                
                for model in models_to_try:
                    try:
                        print(f"Trying model: {model}", file=sys.stderr)
                        response = client.messages.create(
                            model=model,
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
                        print(f"Successfully used model: {model}", file=sys.stderr)
                        return clean_text(summary)
                        
                    except Exception as model_error:
                        print(f"Model {model} failed: {model_error}", file=sys.stderr)
                        continue
                
                print("All models failed, falling back to keyword analysis", file=sys.stderr)
                
            except Exception as ai_error:
                print(f"AI summary failed, falling back to keyword analysis: {ai_error}", file=sys.stderr)
                # Fall through to keyword analysis
        
        # Fallback to keyword analysis
        content_lower = content.lower()
        
        # Extract repository description
        description = ""
        if "repository description:" in content_lower:
            desc_start = content_lower.find("repository description:")
            desc_end = content.find("\n\n", desc_start)
            if desc_end == -1:
                desc_end = len(content)
            description = content[desc_start:desc_end].replace("Repository Description:", "").strip()
        
        # Identify technologies and tools
        technologies = []
        if any(word in content_lower for word in ["python", "py", "pandas", "numpy", "scipy"]):
            technologies.append("Python")
        if any(word in content_lower for word in ["javascript", "js", "node", "react", "vue"]):
            technologies.append("JavaScript")
        if any(word in content_lower for word in ["java", "spring", "maven"]):
            technologies.append("Java")
        if any(word in content_lower for word in ["c++", "cpp", "c plus plus"]):
            technologies.append("C++")
        if any(word in content_lower for word in ["c#", "csharp", "dotnet"]):
            technologies.append("C#")
        if any(word in content_lower for word in ["go", "golang"]):
            technologies.append("Go")
        if any(word in content_lower for word in ["rust"]):
            technologies.append("Rust")
        if any(word in content_lower for word in ["docker", "container"]):
            technologies.append("Docker")
        if any(word in content_lower for word in ["kubernetes", "k8s"]):
            technologies.append("Kubernetes")
        if any(word in content_lower for word in ["machine learning", "ml", "tensorflow", "pytorch"]):
            technologies.append("Machine Learning")
        if any(word in content_lower for word in ["web", "api", "rest", "http"]):
            technologies.append("Web Development")
        
        # Identify topics/categories
        topics = []
        if "topics:" in content_lower:
            topics_start = content_lower.find("topics:")
            topics_end = content.find("\n\n", topics_start)
            if topics_end == -1:
                topics_end = len(content)
            topics_text = content[topics_start:topics_end].replace("Topics:", "").strip()
            topics = [t.strip() for t in topics_text.split(",")]
        
        # Get star count
        stars = ""
        if "stars:" in content_lower:
            stars_start = content_lower.find("stars:")
            stars_end = content.find("\n", stars_start)
            if stars_end == -1:
                stars_end = len(content)
            stars = content[stars_start:stars_end].replace("Stars:", "").strip()
        
        # Get primary language
        language = ""
        if "primary language:" in content_lower:
            lang_start = content_lower.find("primary language:")
            lang_end = content.find("\n", lang_start)
            if lang_end == -1:
                lang_end = len(content)
            language = content[lang_start:lang_end].replace("Primary Language:", "").strip()
        
        # Build summary
        summary_parts = []
        
        if description:
            summary_parts.append(f"Purpose: {description}")
        
        if technologies:
            summary_parts.append(f"Technologies: {', '.join(technologies)}")
        
        if topics:
            summary_parts.append(f"Categories: {', '.join(topics[:5])}")  # Limit to first 5 topics
        
        if language:
            summary_parts.append(f"Primary Language: {language}")
        
        if stars:
            summary_parts.append(f"Popularity: {stars} stars")
        
        # Add general assessment
        if "machine learning" in content_lower or "ai" in content_lower or "neural" in content_lower:
            summary_parts.append("This appears to be an AI/Machine Learning project.")
        elif "web" in content_lower or "api" in content_lower or "frontend" in content_lower:
            summary_parts.append("This appears to be a web development project.")
        elif "data" in content_lower or "analysis" in content_lower or "visualization" in content_lower:
            summary_parts.append("This appears to be a data analysis project.")
        else:
            summary_parts.append("This appears to be a software development project.")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python github_summarizer.py <github_url>")
        print("Example: python github_summarizer.py https://github.com/owner/repo")
        sys.exit(1)
    
    github_url = sys.argv[1]
    
    print(f"Analyzing: {github_url}")
    print("-" * 50)
    
    summary = summarize_github_repo(github_url)
    print(summary)

if __name__ == "__main__":
    main() 