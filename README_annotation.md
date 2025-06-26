# GitHub Repository Annotation Scripts

This directory contains scripts to automatically annotate GitHub repositories with AI-generated summaries using Anthropic's Claude API, including biomedical relevance detection.

## Files

- `annotate_github_repos.py` - Full script to process all repositories in `gits_to_reannotate.tsv`
- `annotate_github_repos_test.py` - Test script to process only the first 5 repositories
- `fix_github_data.py` - Script to clean problematic GitHub data files

## Prerequisites

1. **Anthropic API Key**: You need an Anthropic API key to use the annotation scripts.
   - Get your API key from [Anthropic Console](https://console.anthropic.com/)
   - Set it as an environment variable:
     ```bash
     export ANTHROPIC_API_KEY='your-api-key-here'
     ```

2. **Python Dependencies**: Install required packages:
   ```bash
   pip install pandas requests
   ```

3. **Input File**: Ensure `gits_to_reannotate.tsv` exists in the current directory.

## Usage

### Test Run (Recommended First)

Start with the test script to verify everything works:

```bash
python annotate_github_repos_test.py
```

This will:
- Process only the first 5 repositories
- Show detailed progress and results
- Flag repositories as biomedical or non-biomedical
- Save output to `gits_to_reannotate_test_annotated.tsv`

### Full Run

Once you've verified the test works, run the full annotation:

```bash
python annotate_github_repos.py
```

This will:
- Process all repositories in the input file
- Save progress every 10 repositories
- Track biomedical vs non-biomedical statistics
- Save final output to `gits_to_reannotate_annotated.tsv`

## Output

The scripts add four new columns to the TSV file:

1. **Biomedical Relevance**: Determines if the repository is related to biomedical research, healthcare, neuroscience, or medical applications (YES/NO with explanation)
2. **Code Summary**: Brief summary of the repository's purpose and functionality (less than 300 words)
3. **Data Types**: Description of data types mentioned or used in the repository
4. **Tooling**: Summary of analytics tools, packages, frameworks, and technologies used

## Biomedical Relevance Detection

The scripts now include intelligent detection of biomedical relevance:

- **Biomedical repositories** are flagged with ✅ and include research related to:
  - Healthcare and medical applications
  - Neuroscience and brain research
  - Biomedical data analysis
  - Medical imaging and diagnostics
  - Clinical research tools

- **Non-biomedical repositories** are flagged with ⚠️ and may include:
  - General software tools
  - Non-medical applications
  - Unrelated research areas
  - Personal projects

## Features

- **AI-Powered Analysis**: Uses Anthropic's Claude to generate intelligent summaries
- **Biomedical Detection**: Automatically identifies biomedical vs non-biomedical repositories
- **GitHub Integration**: Fetches README files and repository metadata
- **TSV Format**: Maintains tab-delimited format as requested
- **Error Handling**: Graceful handling of API failures and missing data
- **Rate Limiting**: Respects API rate limits with built-in delays
- **Progress Tracking**: Saves progress every 10 repositories
- **Statistics**: Provides real-time and final statistics on biomedical relevance
- **Content Cleaning**: Ensures output is safe for TSV format

## Troubleshooting

### API Key Issues
- Ensure `ANTHROPIC_API_KEY` is set correctly
- Check that your API key has sufficient credits

### GitHub API Issues
- The scripts use GitHub's public API (no authentication required)
- Some repositories may be private or have restricted access
- Rate limiting may occur with many requests

### File Format Issues
- If you encounter delimiter issues, use `fix_github_data.py` to clean the data first

## Example Output

The annotated file will look like:

```tsv
Study Name	Abbreviation	Diseases Included	Repository Link	Owner	Contributors	Languages	Biomedical Relevance	Code Summary	Data Types	Tooling
ADAMS	ADAMS	Alzheimer's Disease	https://github.com/example/repo	owner	contributor	Python	YES - This repository contains machine learning tools for analyzing brain MRI data	This repository contains a machine learning pipeline for analyzing brain MRI data...	MRI images, clinical data, demographic information	Python, scikit-learn, pandas, numpy, matplotlib
```

## Sample Output

The scripts provide detailed console output showing:

```
Processing 1/5: https://github.com/example/repo
  Retrieved content for owner/repo (1500 characters)
  ✅ BIOMEDICAL: owner/repo
    Relevance: YES - This repository contains machine learning tools for analyzing brain MRI data...
    Code Summary: 245 characters
    Data Types: 89 characters
    Tooling: 156 characters

Summary: 1 out of 5 repositories appear to be non-biomedical
```

## Notes

- The scripts use Claude 3.5 Sonnet for analysis
- Each repository analysis takes approximately 2-3 seconds
- The full dataset may take 30-60 minutes to process
- Results are automatically cleaned to be TSV-safe (no tabs or newlines in content)
- Biomedical relevance detection helps identify repositories that may not belong in a biomedical catalog 