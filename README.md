# CARD Catalogue

A FAIR browser for publicly available and controlled access Alzheimer's disease studies with AI-powered insights.

## Overview

The CARD Catalogue is a comprehensive web application that provides an advanced view of Alzheimer's disease studies and their associated resources. It features four main sections with AI-powered analysis capabilities:

1. **Data**: View and filter studies and datasets from our inventory, with interactive knowledge graphs and AI-powered data analysis
2. **Publications**: Browse and filter publications from PubMed Central with author/affiliation networks and AI-powered publication analysis
3. **Code**: Explore GitHub repositories related to Alzheimer's studies with repository connection graphs and AI-powered code analysis
4. **Biorepositories**: Access iNDI iPSC lines with neurodegenerative disease-associated variants and AI-powered genetic analysis

## Features

### 🔍 **Advanced Data Exploration**
- **Interactive Knowledge Graphs**: Visualize connections between studies, authors, affiliations, and repositories
- **Semantic Search**: Advanced search capabilities across all data types
- **Comprehensive Filtering**: Filter by diseases, data types, tools, genes, variants, and more
- **Data Export**: Export filtered results and graph summaries as CSV files
- **FAIR Compliance Tracking**: Monitor FAIR compliance across studies

### 🤖 **AI-Powered Analysis**
- **Comprehensive AI Analysis**: Multiple analysis types per tab with expert-level insights
- **Research Gap Analysis**: Identify gaps compared to current biomedical research priorities
- **Filtered vs Original Comparisons**: Detailed analysis of how filtering affects research landscape
- **Strategic Recommendations**: Actionable insights for research planning and collaboration
- **Export Analysis Results**: Save AI analysis as timestamped text files

### 📊 **Knowledge Graph Features**
- **Interactive Networks**: Hover for details, zoom, and pan through connections
- **Color-Coded Nodes**: Different colors for different entity types and connection levels
- **Gold Highlights**: Top 3 most connected nodes highlighted for quick identification
- **Detailed Summaries**: Comprehensive metrics and connection analysis
- **Export Capabilities**: Download graph summaries as CSV files

## Live Demo

[Add your Streamlit Cloud deployment URL here]

## Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/card-catalogue.git
   cd card-catalogue
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (optional, for AI analysis):
   ```bash
   # Copy the example secrets file
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   
   # Edit the secrets file with your API key
   # Get your Anthropic API key from: https://console.anthropic.com/
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Set up Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your forked repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Configure secrets** (optional):
   - In your Streamlit Cloud app settings
   - Go to "Secrets" section
   - Add your Anthropic API key:
     ```toml
     ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
     ```

## AI Analysis Features

### **Data Tab Analysis**
- **Basic Data Summary**: Overview of dataset, patterns, and quality assessment
- **Research Gaps Analysis**: Identifies programmatic gaps in research data
- **Filtered vs Original Comparison**: Compares filtered results to the full dataset
- **Knowledge Graph Insights**: Analyzes network structure and collaboration patterns
- **Comprehensive Analysis**: Complete analysis covering all aspects

### **Publications Tab Analysis**
- **Publication Trends Analysis**: Bibliometric analysis and publication patterns
- **Collaboration Network Insights**: Author and affiliation collaboration analysis
- **Research Topic Analysis**: Topic clustering and keyword analysis
- **Author/Affiliation Patterns**: Productivity and institutional analysis
- **Filtered vs Original Comparison**: Detailed comparison of publication datasets
- **Research Gaps vs Current Priorities**: Analysis against 2024-2025 biomedical priorities
- **Comprehensive Publications Analysis**: Complete publication landscape analysis

### **Code Tab Analysis**
- **Repository Technology Analysis**: Programming languages and technology stacks
- **Development Patterns**: Repository structure and development workflows
- **Research Software Trends**: Software categories and ecosystem analysis
- **Collaboration Insights**: Repository collaboration networks
- **Comprehensive Code Analysis**: Complete research software landscape analysis

### **Biorepositories Tab Analysis**
- **Gene Variant Analysis**: Gene distribution and variant patterns
- **Disease Association Patterns**: Disease-gene relationships and mechanisms
- **Biorepository Coverage**: Geographic and population diversity analysis
- **Genetic Diversity Analysis**: Population genetics and variability patterns
- **Filtered vs Original Comparison**: Enhanced comparison with biomedical discovery focus
- **Research Gaps in Neurodegeneration**: Analysis against neurodegeneration research priorities
- **Comprehensive Biorepository Analysis**: Complete genetic research landscape analysis

## Data Sources

The application uses the following data sources:

- **Data Inventory**: Comprehensive list of Alzheimer's studies and datasets with metadata
- **PubMed Central**: Publications related to Alzheimer's studies
- **GitHub**: Code repositories related to Alzheimer's research (biomedical relevance filtered)
- **iNDI Biorepository**: iPSC lines with neurodegenerative disease-associated genetic variants

## Technical Implementation

The application is built using:

- **Streamlit**: Web interface and data visualization
- **Pandas**: Data manipulation and filtering
- **NetworkX**: Knowledge graph creation and analysis
- **Plotly**: Interactive network visualizations
- **scikit-learn**: Semantic search using TF-IDF and cosine similarity
- **Anthropic Claude**: AI-powered analysis and insights (optional)

### Knowledge Graphs

Each tab includes interactive knowledge graphs that show connections between entities:

- **Data Graph**: Connections based on shared diseases, data types, and FAIR compliance
- **Authors Graph**: Co-authorship networks from publications with collaboration patterns
- **Affiliations Graph**: Institutional collaboration networks with co-affiliation patterns
- **Repositories Graph**: Repository connections based on shared content and tools
- **Owners Graph**: Repository owner networks and collaboration patterns

### Semantic Search

The semantic search functionality uses:
1. TF-IDF vectorization of text content
2. Cosine similarity calculation between query and documents
3. Top-k result retrieval based on similarity scores

## File Structure

```
card-catalogue/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .streamlit/
│   ├── secrets.toml               # API keys (not in repo)
│   └── secrets.toml.example       # Example secrets file
├── .gitignore                     # Git ignore rules
└── data/                          # Data files (not in repo)
    ├── studies/
    ├── publications/
    ├── repositories/
    └── biorepositories/
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Anthropic API key for AI analysis (optional)

### Streamlit Configuration

The app uses Streamlit's built-in configuration. Key settings:

- **Page title**: "CARD Catalogue"
- **Page icon**: Custom CARD logo
- **Layout**: Wide layout for better data visualization
- **Theme**: Custom dark theme for better readability

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CARD Initiative**: For providing the study inventory and guidance
- **Alzheimer's Disease Research Community**: For the valuable research data
- **Streamlit**: For the excellent web framework
- **Anthropic**: For the AI analysis capabilities

## Support

For questions or support, please open an issue on GitHub or contact the development team.

## Changelog

### Version 2.0.0
- **Major AI Analysis Enhancement**: Added comprehensive AI-powered analysis to all tabs
- **Enhanced Knowledge Graphs**: Improved visualizations with better color coding and summaries
- **Research Gap Analysis**: Added analysis against current biomedical research priorities
- **Filtered vs Original Comparisons**: Detailed comparison analysis across all tabs
- **Export Enhancements**: Added AI analysis export functionality
- **Biomedical Focus**: Enhanced biorepository analysis with functional validation insights

### Version 1.0.0
- Initial release with all four main tabs (Data, Publications, Code, Biorepositories)
- Interactive knowledge graphs
- Semantic search functionality
- Basic AI-powered explanations
- Comprehensive filtering and export capabilities 