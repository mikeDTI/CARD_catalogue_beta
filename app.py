import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
import glob
import os
import plotly.express as px
import anthropic

# Page configuration
st.set_page_config(
    page_title="CARD Catalogue",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black theme with neon blue highlights
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 4px 4px 0px 0px;
        color: #ffffff;
        border: 1px solid #00bfff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00bfff;
        color: #000000;
    }
    .node-hover {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 4px;
        margin: 5px;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #ffffff;
    }
    .stButton > button {
        background-color: #00bfff;
        color: #000000;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0099cc;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #00bfff;
    }
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #00bfff;
    }
    .stDataFrame {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stSuccess {
        background-color: #00bfff;
        color: #000000;
    }
    .stWarning {
        background-color: #ff6b35;
        color: #ffffff;
    }
    .stError {
        background-color: #ff0000;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

def split_data_modalities(data_modalities_str):
    """
    Split data modalities string into coarse and granular parts.
    
    Format: [coarse_level] granular_details
    Example: [clinical, genetics, imaging] Clinical assessments; MRI; PET
    
    Returns:
        tuple: (coarse_data_types, granular_data_types)
    """
    if pd.isna(data_modalities_str) or not str(data_modalities_str).strip():
        return "", ""
    
    text = str(data_modalities_str).strip()
    
    # Check if there's a bracket pattern
    bracket_match = re.match(r'^\[(.*?)\]\s*(.*)$', text)
    
    if bracket_match:
        coarse_part = bracket_match.group(1).strip()
        granular_part = bracket_match.group(2).strip()
        return coarse_part, granular_part
    else:
        # If no brackets, treat the whole thing as granular
        return "", text

def generate_coarse_statuses(disease_text):
    """
    Generate coarse statuses based on disease/condition text.
    
    Returns:
        str: Comma-separated list of coarse statuses
    """
    if pd.isna(disease_text) or not str(disease_text).strip():
        return ""
    
    text = str(disease_text).lower()
    statuses = []
    
    # Check for key terms
    if any(term in text for term in ['alzheimer', 'ad']):
        statuses.append('Alzheimer')
    if any(term in text for term in ['parkinson', 'pd']):
        statuses.append('Parkinson')
    if 'dementia' in text:
        statuses.append('Dementia')
    if any(term in text for term in ['control', 'normal', 'healthy']):
        statuses.append('Control')
    
    return ', '.join(statuses) if statuses else ""

# Load data
@st.cache_data
def load_data():
    try:
        studies_df = pd.read_csv("dataset-inventory-June_20_2025.tab", sep="\t")
        pubmed_df = pd.read_csv("pubmed_central_20250620_174508.tab", sep="\t")
        github_df = pd.read_csv("gits_to_reannotate_completed_20250626_120254.tsv", sep="\t")
        indi_df = pd.read_csv("iNDI_inventory_20250620_122423.tab", sep="\t")
        
        # Process data modalities for studies and publications
        for df, df_name in [(studies_df, "studies"), (pubmed_df, "publications")]:
            if 'Data Modalities' in df.columns:
                # Split the Data Modalities column
                split_results = df['Data Modalities'].apply(split_data_modalities)
                
                # Create new columns
                df['Coarse Data Types'] = [result[0] for result in split_results]
                df['Granular Data Types'] = [result[1] for result in split_results]
                
                # Reorder columns to put the new columns after Data Modalities
                cols = list(df.columns)
                data_mod_idx = cols.index('Data Modalities')
                
                # Remove the new columns from their current positions
                cols.remove('Coarse Data Types')
                cols.remove('Granular Data Types')
                
                # Insert them after Data Modalities
                cols.insert(data_mod_idx + 1, 'Coarse Data Types')
                cols.insert(data_mod_idx + 2, 'Granular Data Types')
                
                df = df[cols]
        
        # Add General FAIR compliance to studies
        if 'FAIR Compliance Notes' in studies_df.columns:
            studies_df['General FAIR compliance'] = studies_df['FAIR Compliance Notes'].apply(extract_general_fair)
            
            # Reorder columns to put General FAIR compliance before FAIR Compliance Notes
            cols = list(studies_df.columns)
            if 'FAIR Compliance Notes' in cols:
                fair_idx = cols.index('FAIR Compliance Notes')
                cols.remove('General FAIR compliance')
                cols.insert(fair_idx, 'General FAIR compliance')
                studies_df = studies_df[cols]
        
        # Process disease/condition columns for studies, publications, and indi dataframes (not GitHub)
        for df, df_name in [(studies_df, "studies"), (pubmed_df, "publications"), (indi_df, "indi")]:
            # Find disease/condition columns
            disease_cols = []
            if 'Diseases Included' in df.columns:
                disease_cols.append('Diseases Included')
            if 'Condition' in df.columns:
                disease_cols.append('Condition')
            
            for col in disease_cols:
                # Generate coarse statuses
                df[f'Coarse Statuses Included'] = df[col].apply(generate_coarse_statuses)
                
                # Rename original column to Granular Statuses Included
                df.rename(columns={col: 'Granular Statuses Included'}, inplace=True)
                
                # Reorder columns to put Coarse Statuses before Granular Statuses
                cols = list(df.columns)
                if 'Granular Statuses Included' in cols:
                    granular_idx = cols.index('Granular Statuses Included')
                    cols.remove('Coarse Statuses Included')
                    # Insert Coarse Statuses before Granular Statuses
                    cols.insert(granular_idx, 'Coarse Statuses Included')
                    df = df[cols]
        
        # Special handling for GitHub dataframe - keep original "Diseases Included" column
        if 'Diseases Included' in github_df.columns:
            # Don't rename the original column, just add the coarse statuses
            github_df['Coarse Statuses Included'] = github_df['Diseases Included'].apply(generate_coarse_statuses)
            
            # Reorder columns to put Coarse Statuses before Diseases Included
            cols = list(github_df.columns)
            if 'Diseases Included' in cols:
                diseases_idx = cols.index('Diseases Included')
                cols.remove('Coarse Statuses Included')
                # Insert Coarse Statuses before Diseases Included
                cols.insert(diseases_idx, 'Coarse Statuses Included')
                github_df = github_df[cols]
        
        # Additional cleanup for iNDI data - ensure all "0" values are replaced with empty strings
        if 'Granular Statuses Included' in indi_df.columns:
            indi_df['Granular Statuses Included'] = indi_df['Granular Statuses Included'].replace('0', '')
        if 'Coarse Statuses Included' in indi_df.columns:
            indi_df['Coarse Statuses Included'] = indi_df['Coarse Statuses Included'].replace('0', '')
        
        return studies_df, pubmed_df, github_df, indi_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        # Return empty DataFrames with expected columns
        studies_df = pd.DataFrame(columns=['Study Name', 'Abbreviation', 'Data Modalities', 'Coarse Data Types', 'Granular Data Types', 'Granular Statuses Included', 'Coarse Statuses Included', 'FAIR Compliance Notes'])
        pubmed_df = pd.DataFrame(columns=['Study Name', 'Title', 'Authors', 'Abstract', 'Keywords', 'Data Modalities', 'Coarse Data Types', 'Granular Data Types', 'Granular Statuses Included', 'Coarse Statuses Included'])
        github_df = pd.DataFrame(columns=['Study Name', 'Abbreviation', 'Repository Link', 'Languages', 'Tools/Packages', 'Code Summary', 'Granular Statuses Included', 'Coarse Statuses Included'])
        indi_df = pd.DataFrame(columns=['Gene', 'Granular Statuses Included', 'Coarse Statuses Included', 'Genotype', 'Genome Assembly', 'About this gene'])
        return studies_df, pubmed_df, github_df, indi_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrames
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def extract_general_fair(fair_notes):
    """
    Extract the first word from FAIR compliance notes.
    
    Returns:
        str: First word of the FAIR compliance notes, or empty string if none
    """
    if pd.isna(fair_notes) or not str(fair_notes).strip():
        return ""
    
    text = str(fair_notes).strip()
    first_word = text.split()[0] if text else ""
    
    # Return empty string if result would be "0"
    return "" if first_word == "0" else first_word

def perform_text_search(df, query, search_columns):
    """
    Perform simple text search across specified columns.
    
    Args:
        df: DataFrame to search
        query: Search query string
        search_columns: List of column names to search in
    
    Returns:
        DataFrame: Filtered results
    """
    if not query.strip():
        return df
    
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Create a mask for rows that contain the search term
    mask = pd.Series([False] * len(df), index=df.index)
    
    # Search in each specified column
    for col in search_columns:
        if col in df.columns:
            # Convert column to string and search for the query
            mask |= df[col].astype(str).str.contains(query_lower, case=False, na=False)
    
    return df[mask]

def create_studies_knowledge_graph(df):
    """
    Create a knowledge graph for studies data.
    Each study row becomes a node with connections based on shared values in all columns.
    """
    G = nx.Graph()
    
    # Columns to exclude from connections
    exclude_cols = ['Study Name', 'Abbreviation', 'Access URL']
    
    # Add all studies as nodes first
    for idx, row in df.iterrows():
        study_name = row.get('Study Name', 'Unknown Study')
        
        # Create hover text with all column information except excluded ones
        hover_info = []
        for col in df.columns:
            if col not in exclude_cols:
                value = row.get(col, '')
                if pd.notna(value) and str(value).strip():
                    hover_info.append(f"{col}: {str(value).strip()}")
        
        hover_text = "<br>".join(hover_info)
        
        # Add node with hover information
        G.add_node(study_name, 
                  type='study', 
                  size=20,
                  hover_info=hover_text,
                  full_data=row.to_dict())
    
    # Create connections based on shared values in all columns except excluded ones
    for i, row1 in df.iterrows():
        study1 = row1.get('Study Name', 'Unknown Study')
        
        for j, row2 in df.iterrows():
            if i >= j:  # Avoid duplicate connections and self-connections
                continue
                
            study2 = row2.get('Study Name', 'Unknown Study')
            shared_values = []
            
            # Check for shared values in all columns except excluded ones
            for col in df.columns:
                if col not in exclude_cols:
                    val1 = str(row1.get(col, '')).strip()
                    val2 = str(row2.get(col, '')).strip()
                    
                    # Skip empty or NaN values
                    if (pd.notna(row1.get(col)) and pd.notna(row2.get(col)) and 
                        val1 and val2 and val1 != 'nan' and val2 != 'nan'):
                        
                        # For comma-separated values, check if any value is shared
                        if ',' in val1 or ',' in val2:
                            vals1 = [v.strip() for v in val1.split(',') if v.strip()]
                            vals2 = [v.strip() for v in val2.split(',') if v.strip()]
                            shared = set(vals1) & set(vals2)
                            if shared:
                                shared_values.extend(list(shared))
                        # For semicolon-separated values, check if any value is shared
                        elif ';' in val1 or ';' in val2:
                            vals1 = [v.strip() for v in val1.split(';') if v.strip()]
                            vals2 = [v.strip() for v in val2.split(';') if v.strip()]
                            shared = set(vals1) & set(vals2)
                            if shared:
                                shared_values.extend(list(shared))
                        # For simple values, check exact match
                        elif val1 == val2:
                            shared_values.append(val1)
            
            # Add edge if there are shared values
            if shared_values:
                # Create edge label with shared values
                edge_label = f"Shared: {', '.join(set(shared_values))}"
                G.add_edge(study1, study2, 
                          weight=len(set(shared_values)),
                          label=edge_label,
                          shared_values=list(set(shared_values)))
    
    return G

def create_publications_knowledge_graph(df):
    """
    Create a knowledge graph for publications data.
    Each publication row becomes a node with connections based on shared values.
    """
    # This function was replaced with separate authors and affiliations knowledge graphs
    pass

def create_code_knowledge_graph(df):
    """
    Create a knowledge graph for code repositories data.
    Nodes: Repositories, Languages, Tools/Packages, Data Types, Coarse Statuses
    Edges: Connections between repositories and their attributes
    """
    G = nx.Graph()
    
    for _, row in df.iterrows():
        repo_name = row.get('Repository Name', 'Unknown Repository')
        G.add_node(repo_name, type='repository', size=20)
        
        # Add languages
        languages = str(row.get('Languages', '')).split(', ')
        for lang in languages:
            if lang.strip() and lang.strip() != 'nan':
                G.add_node(lang.strip(), type='language', size=15)
                G.add_edge(repo_name, lang.strip())
        
        # Add tools/packages
        tools = str(row.get('Tools/Packages', '')).split(', ')
        for tool in tools[:10]:  # Limit to first 10 tools
            if tool.strip() and tool.strip() != 'nan':
                G.add_node(tool.strip(), type='tool', size=12)
                G.add_edge(repo_name, tool.strip())
        
        # Add data types
        data_types = str(row.get('Data Types/Modalities', '')).split(', ')
        for dt in data_types:
            if dt.strip() and dt.strip() != 'nan':
                G.add_node(dt.strip(), type='data_type', size=15)
                G.add_edge(repo_name, dt.strip())
        
        # Add coarse statuses
        coarse_statuses = str(row.get('Coarse Statuses Included', '')).split(', ')
        for cs in coarse_statuses:
            if cs.strip() and cs.strip() != 'nan':
                G.add_node(cs.strip(), type='status', size=15)
                G.add_edge(repo_name, cs.strip())
    
    return G

def plot_knowledge_graph(G, title, graph_type='default'):
    """
    Create an interactive plotly network visualization of the knowledge graph.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_border_color = []
    node_border_width = []
    
    # Get node degrees for sizing and highlighting
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # Find top 3 most connected nodes
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    top_node_names = [node for node, _ in top_nodes]
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node size based on degree
        degree = degrees.get(node, 0)
        if graph_type in ['authors', 'affiliations']:
            # Smaller nodes for authors and affiliations graphs
            size = max(5, 10 + (degree / max_degree) * 20) if max_degree > 0 else 10
        else:
            # Original sizing for other graphs
            size = max(10, 20 + (degree / max_degree) * 40) if max_degree > 0 else 20
        
        node_size.append(size)
        
        # Node color based on graph type
        if graph_type == 'studies':
            # Color coding for studies based on General FAIR Compliance
            node_attrs = G.nodes[node]
            if 'full_data' in node_attrs:
                fair_compliance = str(node_attrs['full_data'].get('General FAIR compliance', '')).lower()
                if 'excellent' in fair_compliance:
                    color = '#32cd32'  # Green for excellent
                elif 'strong' in fair_compliance:
                    color = '#ffd700'  # Gold for strong
                elif 'good' in fair_compliance:
                    color = '#ff6b35'  # Orange for good
                elif 'fair' in fair_compliance:
                    color = '#ff69b4'  # Pink for fair
                elif 'poor' in fair_compliance:
                    color = '#ff0000'  # Red for poor
                else:
                    color = '#00bfff'  # Blue for unknown/other
            else:
                color = '#00bfff'  # Blue for unknown
        elif graph_type == 'authors':
            # Color authors by publication count (we'll use degree as proxy)
            if degree >= max_degree * 0.8:
                color = '#ff7f0e'  # Orange for highly connected
            elif degree >= max_degree * 0.5:
                color = '#2ca02c'  # Green for moderately connected
            else:
                color = '#1f77b4'  # Blue for less connected
        elif graph_type == 'affiliations':
            # Color affiliations by activity level
            if degree >= max_degree * 0.8:
                color = '#d62728'  # Red for highly active
            elif degree >= max_degree * 0.5:
                color = '#9467bd'  # Purple for moderately active
            else:
                color = '#8c564b'  # Brown for less active
        elif graph_type in ['repositories', 'owners']:
            # Color repositories and owners by connection level
            if degree >= max_degree * 0.8:
                color = '#ff7f0e'  # Orange for highly connected
            elif degree >= max_degree * 0.5:
                color = '#2ca02c'  # Green for moderately connected
            else:
                color = '#1f77b4'  # Blue for less connected
        else:
            # Default color scheme for other graphs
            if 'FAIR' in str(node).upper():
                color = '#2ca02c'  # Green for FAIR-related
            else:
                color = '#1f77b4'  # Blue for others
        
        node_color.append(color)
        
        # Border for top connected nodes
        if node in top_node_names:
            node_border_color.append('gold')
            node_border_width.append(3)
        else:
            node_border_color.append('black')
            node_border_width.append(1)
        
        # Node text/hover information
        if graph_type == 'authors':
            node_text.append(f"{node}")
        elif graph_type == 'affiliations':
            node_text.append(f"{node}")
        else:
            node_text.append(f"{node}")
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge weight information
        weight = G[edge[0]][edge[1]].get('weight', 1)
        if graph_type == 'authors':
            edge_text.append(f"Co-authorship: {weight} shared publications")
        elif graph_type == 'affiliations':
            edge_text.append(f"Co-affiliation: {weight} shared publications")
        else:
            edge_text.append(f"Weight: {weight}")
    
    # Create the network plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
        name='Connections'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(
                color=node_border_color,
                width=node_border_width
            ),
            opacity=0.8
        ),
        showlegend=True,
        name='Nodes'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Hover over nodes for details",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(size=10, color='white')
        ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white')
        )
    )
    
    return fig

def create_repositories_knowledge_graph(df):
    """
    Create a knowledge graph for repositories from code data.
    Each repository becomes a node with connections based on shared languages, code summary, and data types.
    """
    G = nx.Graph()
    
    # Add all repositories as nodes first
    for idx, row in df.iterrows():
        repo_name = row.get('Repository Name', 'Unknown Repository')
        repo_url = row.get('Repository Link', '')
        
        # Create hover text with repository information
        hover_info = []
        hover_info.append(f"Repository: {repo_name}")
        if repo_url:
            hover_info.append(f"URL: {repo_url}")
        
        # Add other relevant information
        for col in ['Languages', 'Code Summary', 'Data Types/Modalities']:
            if col in df.columns:
                value = row.get(col, '')
                if pd.notna(value) and str(value).strip():
                    hover_info.append(f"{col}: {str(value).strip()}")
        
        hover_text = "<br>".join(hover_info)
        
        # Add node with hover information
        G.add_node(repo_name, 
                  type='repository', 
                  size=20,
                  hover_info=hover_text,
                  full_data=row.to_dict())
    
    # Create connections based on shared values in Languages, Code Summary, and Data Types/Modalities
    for i, row1 in df.iterrows():
        repo1 = row1.get('Repository Name', 'Unknown Repository')
        
        for j, row2 in df.iterrows():
            if i >= j:  # Avoid duplicate connections and self-connections
                continue
                
            repo2 = row2.get('Repository Name', 'Unknown Repository')
            shared_values = []
            
            # Check for shared values in specified columns
            connection_cols = ['Languages', 'Code Summary', 'Data Types/Modalities']
            
            for col in connection_cols:
                if col in df.columns:
                    val1 = str(row1.get(col, '')).strip()
                    val2 = str(row2.get(col, '')).strip()
                    
                    # Skip empty or NaN values
                    if (pd.notna(row1.get(col)) and pd.notna(row2.get(col)) and 
                        val1 and val2 and val1 != 'nan' and val2 != 'nan'):
                        
                        # For comma-separated values (Languages, Data Types), check if any value is shared
                        if col in ['Languages', 'Data Types/Modalities'] and (',' in val1 or ',' in val2):
                            vals1 = [v.strip() for v in val1.split(',') if v.strip()]
                            vals2 = [v.strip() for v in val2.split(',') if v.strip()]
                            shared = set(vals1) & set(vals2)
                            if shared:
                                shared_values.extend(list(shared))
                        # For Code Summary, check for shared words (case-insensitive)
                        elif col == 'Code Summary':
                            # Convert to lowercase and split into words
                            words1 = set(val1.lower().split())
                            words2 = set(val2.lower().split())
                            # Filter out common stop words and short words
                            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
                            words1 = {word for word in words1 if len(word) > 3 and word not in stop_words}
                            words2 = {word for word in words2 if len(word) > 3 and word not in stop_words}
                            shared = words1 & words2
                            if len(shared) >= 2:  # Require at least 2 shared words
                                shared_values.extend(list(shared)[:3])  # Limit to first 3 shared words
            
            # Add edge if there are shared values
            if shared_values:
                # Create edge label with shared values
                edge_label = f"Shared: {', '.join(set(shared_values))}"
                G.add_edge(repo1, repo2, 
                          weight=len(set(shared_values)),
                          label=edge_label,
                          shared_values=list(set(shared_values)))
    
    return G

def create_owners_knowledge_graph(df):
    """
    Create a knowledge graph for repository owners from code data.
    Each owner becomes a node with connections based on shared languages, code summary, and data types.
    """
    G = nx.Graph()
    
    # Collect all owners and their repositories
    owner_repositories = {}
    
    for idx, row in df.iterrows():
        owner = row.get('Owner', 'Unknown Owner')
        if pd.notna(owner) and str(owner).strip():
            owner = str(owner).strip()
            
            # Add owner to the graph
            if owner not in G.nodes():
                G.add_node(owner, type='owner', size=15)
            
            # Track repositories for each owner
            if owner not in owner_repositories:
                owner_repositories[owner] = []
            owner_repositories[owner].append(idx)
    
    # Create connections between owners based on shared characteristics
    for owner1 in owner_repositories:
        for owner2 in owner_repositories:
            if owner1 < owner2:  # Avoid duplicate connections
                shared_values = []
                
                # Get all repositories for each owner
                repos1 = owner_repositories[owner1]
                repos2 = owner_repositories[owner2]
                
                # Check for shared values in specified columns across all repositories
                connection_cols = ['Languages', 'Code Summary', 'Data Types/Modalities']
                
                for col in connection_cols:
                    if col in df.columns:
                        # Collect all values for each owner
                        vals1 = set()
                        vals2 = set()
                        
                        for repo_idx in repos1:
                            val = str(df.iloc[repo_idx].get(col, '')).strip()
                            if pd.notna(val) and val and val != 'nan':
                                if col in ['Languages', 'Data Types/Modalities'] and ',' in val:
                                    vals1.update([v.strip() for v in val.split(',') if v.strip()])
                                else:
                                    vals1.add(val)
                        
                        for repo_idx in repos2:
                            val = str(df.iloc[repo_idx].get(col, '')).strip()
                            if pd.notna(val) and val and val != 'nan':
                                if col in ['Languages', 'Data Types/Modalities'] and ',' in val:
                                    vals2.update([v.strip() for v in val.split(',') if v.strip()])
                                else:
                                    vals2.add(val)
                        
                        # Find shared values
                        shared = vals1 & vals2
                        if shared:
                            shared_values.extend(list(shared))
                
                # Add edge if there are shared values
                if shared_values:
                    # Create edge with weight based on number of shared values
                    G.add_edge(owner1, owner2, 
                              weight=len(set(shared_values)),
                              shared_values=list(set(shared_values)))
    
    return G

def create_knowledge_graph_summary(G, df):
    """
    Create a summary table of the knowledge graph statistics.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate basic statistics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Find most connected nodes
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]  # Top 3 only
    
    # Find least connected nodes
    least_connected = sorted(degrees.items(), key=lambda x: x[1])[:3]  # Bottom 3
    
    # Analyze connections by General FAIR Compliance
    connection_summary = {}
    for node in G.nodes():
        node_attrs = G.nodes[node]
        if 'full_data' in node_attrs:
            fair_compliance = str(node_attrs['full_data'].get('General FAIR compliance', '')).lower()
            degree = degrees.get(node, 0)
            
            if 'excellent' in fair_compliance:
                category = 'Excellent'
            elif 'strong' in fair_compliance:
                category = 'Strong'
            elif 'good' in fair_compliance:
                category = 'Good'
            elif 'fair' in fair_compliance:
                category = 'Fair'
            elif 'poor' in fair_compliance:
                category = 'Poor'
            else:
                category = 'Unknown'
            
            if category not in connection_summary:
                connection_summary[category] = {'count': 0, 'total_connections': 0}
            connection_summary[category]['count'] += 1
            connection_summary[category]['total_connections'] += degree
    
    # Analyze top Coarse Data Types
    coarse_data_counts = {}
    for node in G.nodes():
        node_attrs = G.nodes[node]
        if 'full_data' in node_attrs:
            coarse_types = str(node_attrs['full_data'].get('Coarse Data Types', '')).split(', ')
            for ct in coarse_types:
                if ct.strip() and ct.strip() != 'nan':
                    if ct.strip() not in coarse_data_counts:
                        coarse_data_counts[ct.strip()] = 0
                    coarse_data_counts[ct.strip()] += 1
    
    top_coarse_types = sorted(coarse_data_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    least_coarse_types = sorted(coarse_data_counts.items(), key=lambda x: x[1])[:3]  # Bottom 3
    
    # Create summary dataframe
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Studies',
        'Value': len(G.nodes())
    })
    summary_data.append({
        'Metric': 'Total Connections',
        'Value': len(G.edges())
    })
    summary_data.append({
        'Metric': 'Average Connections per Study',
        'Value': f"{avg_degree:.1f}"
    })
    summary_data.append({
        'Metric': 'Most Connected Study',
        'Value': f"{most_connected[0][0]} ({most_connected[0][1]} connections)" if most_connected else "N/A"
    })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # FAIR Rating Breakdown
    summary_data.append({
        'Metric': 'FAIR Rating Breakdown',
        'Value': ''
    })
    
    for category, stats in connection_summary.items():
        avg_conn = stats['total_connections'] / stats['count'] if stats['count'] > 0 else 0
        summary_data.append({
            'Metric': f"{category} Studies",
            'Value': f"{stats['count']} (avg {avg_conn:.1f} connections)"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top Coarse Data Types
    summary_data.append({
        'Metric': 'Top 3 Coarse Data Types',
        'Value': ''
    })
    
    for i, (coarse_type, count) in enumerate(top_coarse_types, 1):
        summary_data.append({
            'Metric': f"{i}. {coarse_type}",
            'Value': f"{count} studies"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least Common Coarse Data Types
    summary_data.append({
        'Metric': 'Least Common Coarse Data Types',
        'Value': ''
    })
    
    for i, (coarse_type, count) in enumerate(least_coarse_types, 1):
        summary_data.append({
            'Metric': f"{i}. {coarse_type}",
            'Value': f"{count} studies"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top connected studies
    summary_data.append({
        'Metric': 'Top 3 Most Connected Studies',
        'Value': ''
    })
    
    for i, (study, degree) in enumerate(most_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {study}",
            'Value': f"{degree} connections"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least connected studies
    summary_data.append({
        'Metric': 'Least Connected Studies',
        'Value': ''
    })
    
    for i, (study, degree) in enumerate(least_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {study}",
            'Value': f"{degree} connections"
        })
    
    return pd.DataFrame(summary_data)

def create_authors_knowledge_graph(df):
    """
    Create a knowledge graph for authors from publications data.
    Each author becomes a node with connections based on co-authorship.
    """
    G = nx.Graph()
    
    # Collect all authors and their publications
    author_publications = {}
    
    for idx, row in df.iterrows():
        authors_str = row.get('Authors', '')
        if pd.notna(authors_str) and str(authors_str).strip():
            authors = [author.strip() for author in str(authors_str).split(';') if author.strip()]
            
            # Add authors to the graph
            for author in authors:
                if author not in G.nodes():
                    G.add_node(author, type='author', size=15)
                
                # Track publications for each author
                if author not in author_publications:
                    author_publications[author] = []
                author_publications[author].append(idx)
    
    # Create connections between co-authors
    for author1 in author_publications:
        for author2 in author_publications:
            if author1 < author2:  # Avoid duplicate connections
                # Find publications where both authors appear
                pubs1 = set(author_publications[author1])
                pubs2 = set(author_publications[author2])
                shared_pubs = pubs1 & pubs2
                
                if shared_pubs:
                    # Create edge with weight based on number of shared publications
                    G.add_edge(author1, author2, 
                              weight=len(shared_pubs),
                              shared_publications=list(shared_pubs))
    
    return G

def create_affiliations_knowledge_graph(df):
    """
    Create a knowledge graph for affiliations from publications data.
    Each affiliation becomes a node with connections based on co-affiliation.
    """
    G = nx.Graph()
    
    # Collect all affiliations and their publications
    affiliation_publications = {}
    
    for idx, row in df.iterrows():
        affiliations_str = row.get('Affiliations', '')
        if pd.notna(affiliations_str) and str(affiliations_str).strip():
            affiliations = [aff.strip() for aff in str(affiliations_str).split(';') if aff.strip()]
            
            # Add affiliations to the graph
            for affiliation in affiliations:
                if affiliation not in G.nodes():
                    G.add_node(affiliation, type='affiliation', size=15)
                
                # Track publications for each affiliation
                if affiliation not in affiliation_publications:
                    affiliation_publications[affiliation] = []
                affiliation_publications[affiliation].append(idx)
    
    # Create connections between co-affiliated institutions
    for aff1 in affiliation_publications:
        for aff2 in affiliation_publications:
            if aff1 < aff2:  # Avoid duplicate connections
                # Find publications where both affiliations appear
                pubs1 = set(affiliation_publications[aff1])
                pubs2 = set(affiliation_publications[aff2])
                shared_pubs = pubs1 & pubs2
                
                if shared_pubs:
                    # Create edge with weight based on number of shared publications
                    G.add_edge(aff1, aff2, 
                              weight=len(shared_pubs),
                              shared_publications=list(shared_pubs))
    
    return G

def create_authors_knowledge_graph_summary(G, df):
    """
    Create a summary table of the authors knowledge graph statistics.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate basic statistics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Find most connected authors
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Find least connected authors
    least_connected = sorted(degrees.items(), key=lambda x: x[1])[:3]
    
    # Count publications per author
    author_pub_counts = {}
    for idx, row in df.iterrows():
        authors_str = row.get('Authors', '')
        if pd.notna(authors_str) and str(authors_str).strip():
            authors = [author.strip() for author in str(authors_str).split(';') if author.strip()]
            for author in authors:
                if author not in author_pub_counts:
                    author_pub_counts[author] = 0
                author_pub_counts[author] += 1
    
    # Find most and least prolific authors
    most_prolific = sorted(author_pub_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    least_prolific = sorted(author_pub_counts.items(), key=lambda x: x[1])[:3]
    
    # Create summary dataframe
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Authors',
        'Value': len(G.nodes())
    })
    summary_data.append({
        'Metric': 'Total Co-authorship Connections',
        'Value': len(G.edges())
    })
    summary_data.append({
        'Metric': 'Average Connections per Author',
        'Value': f"{avg_degree:.1f}"
    })
    summary_data.append({
        'Metric': 'Most Connected Author',
        'Value': f"{most_connected[0][0]} ({most_connected[0][1]} connections)" if most_connected else "N/A"
    })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Most Prolific Authors
    summary_data.append({
        'Metric': 'Most Prolific Authors',
        'Value': ''
    })
    
    for i, (author, count) in enumerate(most_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {author}",
            'Value': f"{count} publications"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least Prolific Authors
    summary_data.append({
        'Metric': 'Least Prolific Authors',
        'Value': ''
    })
    
    for i, (author, count) in enumerate(least_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {author}",
            'Value': f"{count} publications"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top connected authors
    summary_data.append({
        'Metric': 'Top 3 Most Connected Authors',
        'Value': ''
    })
    
    for i, (author, degree) in enumerate(most_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {author}",
            'Value': f"{degree} co-authorship connections"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least connected authors
    summary_data.append({
        'Metric': 'Least Connected Authors',
        'Value': ''
    })
    
    for i, (author, degree) in enumerate(least_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {author}",
            'Value': f"{degree} co-authorship connections"
        })
    
    return pd.DataFrame(summary_data)

def create_affiliations_knowledge_graph_summary(G, df):
    """
    Create a summary table of the affiliations knowledge graph statistics.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate basic statistics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Find most connected affiliations
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Find least connected affiliations
    least_connected = sorted(degrees.items(), key=lambda x: x[1])[:3]
    
    # Count publications per affiliation
    affiliation_pub_counts = {}
    for idx, row in df.iterrows():
        affiliations_str = row.get('Affiliations', '')
        if pd.notna(affiliations_str) and str(affiliations_str).strip():
            affiliations = [aff.strip() for aff in str(affiliations_str).split(';') if aff.strip()]
            for affiliation in affiliations:
                if affiliation not in affiliation_pub_counts:
                    affiliation_pub_counts[affiliation] = 0
                affiliation_pub_counts[affiliation] += 1
    
    # Find most and least active affiliations
    most_active = sorted(affiliation_pub_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    least_active = sorted(affiliation_pub_counts.items(), key=lambda x: x[1])[:3]
    
    # Create summary dataframe
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Affiliations',
        'Value': len(G.nodes())
    })
    summary_data.append({
        'Metric': 'Total Co-affiliation Connections',
        'Value': len(G.edges())
    })
    summary_data.append({
        'Metric': 'Average Connections per Affiliation',
        'Value': f"{avg_degree:.1f}"
    })
    summary_data.append({
        'Metric': 'Most Connected Affiliation',
        'Value': f"{most_connected[0][0]} ({most_connected[0][1]} connections)" if most_connected else "N/A"
    })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Most Active Affiliations
    summary_data.append({
        'Metric': 'Most Active Affiliations',
        'Value': ''
    })
    
    for i, (affiliation, count) in enumerate(most_active, 1):
        summary_data.append({
            'Metric': f"{i}. {affiliation}",
            'Value': f"{count} publications"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least Active Affiliations
    summary_data.append({
        'Metric': 'Least Active Affiliations',
        'Value': ''
    })
    
    for i, (affiliation, count) in enumerate(least_active, 1):
        summary_data.append({
            'Metric': f"{i}. {affiliation}",
            'Value': f"{count} publications"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top connected affiliations
    summary_data.append({
        'Metric': 'Top 3 Most Connected Affiliations',
        'Value': ''
    })
    
    for i, (affiliation, degree) in enumerate(most_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {affiliation}",
            'Value': f"{degree} co-affiliation connections"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least connected affiliations
    summary_data.append({
        'Metric': 'Least Connected Affiliations',
        'Value': ''
    })
    
    for i, (affiliation, degree) in enumerate(least_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {affiliation}",
            'Value': f"{degree} co-affiliation connections"
        })
    
    return pd.DataFrame(summary_data)

def create_repositories_knowledge_graph_summary(G, df):
    """
    Create a summary table of the repositories knowledge graph statistics.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate basic statistics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Find most connected repositories
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Find least connected repositories
    least_connected = sorted(degrees.items(), key=lambda x: x[1])[:3]
    
    # Count repositories per owner
    owner_repo_counts = {}
    for idx, row in df.iterrows():
        owner = row.get('Owner', 'Unknown Owner')
        if pd.notna(owner) and str(owner).strip():
            owner = str(owner).strip()
            if owner not in owner_repo_counts:
                owner_repo_counts[owner] = 0
            owner_repo_counts[owner] += 1
    
    # Find most and least prolific owners
    most_prolific = sorted(owner_repo_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    least_prolific = sorted(owner_repo_counts.items(), key=lambda x: x[1])[:3]
    
    # Create summary dataframe
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Repositories',
        'Value': len(G.nodes())
    })
    summary_data.append({
        'Metric': 'Total Connections',
        'Value': len(G.edges())
    })
    summary_data.append({
        'Metric': 'Average Connections per Repository',
        'Value': f"{avg_degree:.1f}"
    })
    summary_data.append({
        'Metric': 'Most Connected Repository',
        'Value': f"{most_connected[0][0]} ({most_connected[0][1]} connections)" if most_connected else "N/A"
    })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Most Prolific Owners
    summary_data.append({
        'Metric': 'Most Prolific Owners',
        'Value': ''
    })
    
    for i, (owner, count) in enumerate(most_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{count} repositories"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least Prolific Owners
    summary_data.append({
        'Metric': 'Least Prolific Owners',
        'Value': ''
    })
    
    for i, (owner, count) in enumerate(least_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{count} repositories"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top connected repositories
    summary_data.append({
        'Metric': 'Top 3 Most Connected Repositories',
        'Value': ''
    })
    
    for i, (repo, degree) in enumerate(most_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {repo}",
            'Value': f"{degree} connections"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least connected repositories
    summary_data.append({
        'Metric': 'Least Connected Repositories',
        'Value': ''
    })
    
    for i, (repo, degree) in enumerate(least_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {repo}",
            'Value': f"{degree} connections"
        })
    
    return pd.DataFrame(summary_data)

def create_owners_knowledge_graph_summary(G, df):
    """
    Create a summary table of the owners knowledge graph statistics.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Calculate basic statistics
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    # Find most connected owners
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Find least connected owners
    least_connected = sorted(degrees.items(), key=lambda x: x[1])[:3]
    
    # Count repositories per owner
    owner_repo_counts = {}
    for idx, row in df.iterrows():
        owner = row.get('Owner', 'Unknown Owner')
        if pd.notna(owner) and str(owner).strip():
            owner = str(owner).strip()
            if owner not in owner_repo_counts:
                owner_repo_counts[owner] = 0
            owner_repo_counts[owner] += 1
    
    # Find most and least prolific owners
    most_prolific = sorted(owner_repo_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    least_prolific = sorted(owner_repo_counts.items(), key=lambda x: x[1])[:3]
    
    # Create summary dataframe
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Owners',
        'Value': len(G.nodes())
    })
    summary_data.append({
        'Metric': 'Total Connections',
        'Value': len(G.edges())
    })
    summary_data.append({
        'Metric': 'Average Connections per Owner',
        'Value': f"{avg_degree:.1f}"
    })
    summary_data.append({
        'Metric': 'Most Connected Owner',
        'Value': f"{most_connected[0][0]} ({most_connected[0][1]} connections)" if most_connected else "N/A"
    })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Most Prolific Owners
    summary_data.append({
        'Metric': 'Most Prolific Owners',
        'Value': ''
    })
    
    for i, (owner, count) in enumerate(most_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{count} repositories"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least Prolific Owners
    summary_data.append({
        'Metric': 'Least Prolific Owners',
        'Value': ''
    })
    
    for i, (owner, count) in enumerate(least_prolific, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{count} repositories"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Top connected owners
    summary_data.append({
        'Metric': 'Top 3 Most Connected Owners',
        'Value': ''
    })
    
    for i, (owner, degree) in enumerate(most_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{degree} connections"
        })
    
    # Add empty row
    summary_data.append({
        'Metric': '',
        'Value': ''
    })
    
    # Least connected owners
    summary_data.append({
        'Metric': 'Least Connected Owners',
        'Value': ''
    })
    
    for i, (owner, degree) in enumerate(least_connected, 1):
        summary_data.append({
            'Metric': f"{i}. {owner}",
            'Value': f"{degree} connections"
        })
    
    return pd.DataFrame(summary_data)

# Header with logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("card_logo.png", width=150)
with col2:
    st.title("CARD Catalogue")
    st.markdown("A FAIR browser for publicly available and controlled access Alzheimer's disease studies.")

# Create tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Data",
    "Publications", 
    "Code",
    "Biorepositories",
    "About"
])

# Tab 0: Data
with tab0:
    studies_df, _, _, _ = load_data()
    
    st.markdown("""
    ### How this list was generated:
    This inventory represents a comprehensive collection of major Alzheimer's disease studies and datasets,
    including information about data modalities, sample sizes, and FAIR compliance.
    """)
    
    # Search
    search_query = st.text_input(
        "Search data (searches across all columns)",
        key="studies_search"
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        # Coarse Data Types filter
        coarse_types = st.multiselect(
            "Filter by Coarse Data Types",
            options=sorted(set(ct for cts in studies_df["Coarse Data Types"].dropna() 
                             for ct in str(cts).split(", ") if ct.strip())),
            default=[],
            key="studies_coarse_types"
        )
        
        # Granular Data Types filter
        granular_types = st.multiselect(
            "Filter by Granular Data Types",
            options=sorted(set(gt for gts in studies_df["Granular Data Types"].dropna() 
                             for gt in str(gts).split("; ") if gt.strip())),
            default=[],
            key="studies_granular_types"
        )
        
        # Granular Statuses filter
        granular_statuses = st.multiselect(
            "Filter by Granular Statuses",
            options=sorted(set(gs for gss in studies_df["Granular Statuses Included"].dropna() for gs in gss.split("; "))),
            default=[],
            key="studies_granular_statuses"
        )
    with col2:
        # Coarse Statuses filter
        coarse_statuses = st.multiselect(
            "Filter by Coarse Statuses",
            options=sorted(set(cs for css in studies_df["Coarse Statuses Included"].dropna() for cs in css.split(", ") if cs.strip())),
            default=[],
            key="studies_coarse_statuses"
        )
        
        # FAIR Compliance filter
        fair_levels = st.multiselect(
            "Filter by General FAIR Compliance",
            options=sorted([str(x) for x in studies_df["General FAIR compliance"].dropna().unique()]),
            default=[],
            key="studies_fair"
        )
    
    # Filter data
    filtered_df = studies_df.copy()
    if search_query:
        filtered_df = perform_text_search(filtered_df, search_query, list(filtered_df.columns))
    if coarse_types:
        filtered_df = filtered_df[filtered_df["Coarse Data Types"].apply(lambda x: any(ct in str(x).split(", ") for ct in coarse_types))]
    if granular_types:
        filtered_df = filtered_df[filtered_df["Granular Data Types"].apply(lambda x: any(gt in str(x).split("; ") for gt in granular_types))]
    if granular_statuses:
        filtered_df = filtered_df[filtered_df["Granular Statuses Included"].apply(lambda x: any(gs in str(x).split("; ") for gs in granular_statuses))]
    if coarse_statuses:
        filtered_df = filtered_df[filtered_df["Coarse Statuses Included"].apply(lambda x: any(cs in str(x).split(", ") for cs in coarse_statuses))]
    if fair_levels:
        filtered_df = filtered_df[filtered_df["General FAIR compliance"].astype(str).isin(fair_levels)]
    
    # Display filtered data
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data matches the current filters.")

    # Knowledge Graph Buttons
    if st.button("Show Knowledge Graph", key="studies_graph"):
        st.markdown("### Knowledge Graph")
        st.markdown("Interactive network visualization showing connections between datasets, data types, disease statuses, and FAIR compliance levels. Most connected nodes are highlighted in gold.")
        
        # Create and display knowledge graph using filtered data
        if len(filtered_df) > 0:
            G = create_studies_knowledge_graph(filtered_df)
            fig = plot_knowledge_graph(G, "Data Knowledge Graph", graph_type='studies')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Create and display summary table
            st.markdown("### 📊 Graph Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Datasets", len(G.nodes()))
            with col2:
                st.metric("Total Connections", len(G.edges()))
            with col3:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Connections per Dataset", f"{avg_degree:.1f}")
            with col4:
                fair_excellent = sum(1 for node in G.nodes() if 'full_data' in G.nodes[node] and 
                                   'excellent' in str(G.nodes[node]['full_data'].get('General FAIR compliance', '')).lower())
                st.metric("Excellent FAIR Compliance", fair_excellent)
            
            # Show most connected studies
            if len(G.nodes()) > 0:
                degrees = dict(G.degree())
                top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.markdown("#### 🔗 Most Connected Datasets")
                for study, degree in top_connected:
                    st.write(f"• **{study}** - {degree} connections")
            
            # Detailed summary table
            summary_df = create_knowledge_graph_summary(G, filtered_df)
            if summary_df is not None:
                st.markdown("#### 📋 Detailed Summary")
                st.dataframe(summary_df, use_container_width=True)
                
                # Export summary option
                if st.button("Export Summary as CSV", key="studies_summary_export"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data_knowledge_graph_summary_{timestamp}.csv"
                    summary_df.to_csv(filename, index=False)
                    st.success(f"Summary exported to {filename}")
        else:
            st.warning("No data available to create knowledge graph.")

    # AI-Powered Analysis
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Data Analysis")
    st.markdown("Get AI insights about your data, research gaps, and recommendations for improvement.")
    
    # Check if Anthropic API key is available
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    
    if not anthropic_api_key:
        st.info("💡 **AI Analysis Available**: Add your Anthropic API key to `.streamlit/secrets.toml` to enable AI-powered insights.")
        st.code("ANTHROPIC_API_KEY = 'your-api-key-here'")
    else:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["Basic Data Summary", "Research Gaps Analysis", "Filtered vs Original Comparison", "Knowledge Graph Insights", "Comprehensive Analysis"],
            key="analysis_type"
        )
        
        if st.button("Generate AI Analysis", key="ai_analysis"):
            with st.spinner("🤖 Claude is analyzing your data..."):
                try:
                    # Prepare data summary for analysis
                    data_summary = f"""
                    Original dataset: {len(studies_df)} rows, {len(studies_df.columns)} columns
                    Filtered dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns
                    
                    Original columns: {list(studies_df.columns)}
                    Filtered columns: {list(filtered_df.columns)}
                    
                    Sample of original data (first 3 rows):
                    {studies_df.head(3).to_string()}
                    
                    Sample of filtered data (first 3 rows):
                    {filtered_df.head(3).to_string()}
                    """
                    
                    # Add knowledge graph info if available
                    kg_info = ""
                    if 'G' in locals() and len(G.nodes()) > 0:
                        degrees = dict(G.degree())
                        top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                        kg_info = f"""
                        Knowledge Graph Summary:
                        - Total nodes: {len(G.nodes())}
                        - Total connections: {len(G.edges())}
                        - Average connections per node: {sum(degrees.values()) / len(degrees):.1f}
                        - Top 5 most connected nodes: {top_connected}
                        """
                    
                    # Create prompts based on analysis type
                    if analysis_type == "Basic Data Summary":
                        prompt = f"""
                        You are an expert data analyst specializing in Alzheimer's disease research data. 
                        Analyze this dataset and provide a comprehensive summary.
                        
                        {data_summary}
                        
                        Please provide:
                        1. A clear summary of what this dataset contains
                        2. Key patterns or trends you observe
                        3. Data quality assessment
                        4. Notable strengths of the dataset
                        5. Potential limitations or areas for improvement
                        
                        Format your response in clear sections with bullet points where appropriate.
                        """
                    
                    elif analysis_type == "Research Gaps Analysis":
                        prompt = f"""
                        You are an expert research analyst specializing in Alzheimer's disease research. 
                        Analyze this dataset to identify programmatic gaps in research data.
                        
                        {data_summary}
                        
                        Please identify:
                        1. **Data Coverage Gaps**: What types of data or populations are missing?
                        2. **Methodological Gaps**: What research methods or approaches are underrepresented?
                        3. **Temporal Gaps**: Are there time periods or longitudinal aspects missing?
                        4. **Demographic Gaps**: Are certain populations underrepresented?
                        5. **Technical Gaps**: What technologies or platforms are missing?
                        6. **Collaboration Gaps**: Are there opportunities for cross-study collaboration?
                        7. **Recommendations**: How could these gaps be addressed in future research?
                        
                        Be specific and provide actionable insights.
                        """
                    
                    elif analysis_type == "Filtered vs Original Comparison":
                        prompt = f"""
                        You are an expert data analyst specializing in research publication analysis. 
                        Compare the filtered publications dataset to the original publications dataset.
                        
                        {data_summary}
                        
                        Please provide a detailed comparison analysis:
                        
                        1. **Data Reduction Analysis**
                           - How many publications were filtered out (percentage and absolute numbers)?
                           - What specific filters caused the most data reduction?
                           - Which study names, data types, or keywords were most commonly filtered?
                        
                        2. **Content Pattern Changes**
                           - How do the research topics differ between original and filtered datasets?
                           - What changes occur in author collaboration patterns?
                           - How do institutional affiliations change between datasets?
                           - What changes are evident in publication dates or journals?
                        
                        3. **Bias and Representativeness Assessment**
                           - Are there any systematic biases introduced by the filtering?
                           - How representative is the filtered subset of the broader research landscape?
                           - What types of research might be underrepresented in the filtered results?
                           - Are certain institutions or authors disproportionately affected?
                        
                        4. **Research Priority Implications**
                           - What does the filtering reveal about current research priorities?
                           - Which research areas are being emphasized or de-emphasized?
                           - How do the filtered results align with Alzheimer's disease research priorities?
                        
                        5. **Quality and Impact Assessment**
                           - How does the quality of publications change between datasets?
                           - Are high-impact publications more or less likely to be filtered?
                           - What does this tell us about research quality vs. quantity?
                        
                        6. **Strategic Recommendations**
                           - How could the filtering criteria be improved?
                           - What research areas might need more attention?
                           - How could the research process be enhanced to address gaps?
                        
                        Provide specific examples from the data to support your analysis.
                        """
                    
                    elif analysis_type == "Knowledge Graph Insights":
                        if kg_info:
                            prompt = f"""
                            You are an expert in network analysis and research collaboration patterns. 
                            Analyze this knowledge graph of Alzheimer's disease research data.
                            
                            {data_summary}
                            {kg_info}
                            
                            Please provide:
                            1. **Network Structure Analysis**: What does the graph structure tell us?
                            2. **Collaboration Patterns**: What collaboration patterns emerge?
                            3. **Research Clusters**: Are there distinct research communities?
                            4. **Centrality Insights**: What do the most connected nodes represent?
                            5. **Gap Identification**: What connections are missing?
                            6. **Strategic Recommendations**: How could collaboration be improved?
                            
                            Focus on actionable insights for research coordination.
                            """
                        else:
                            st.warning("No knowledge graph available. Please generate a knowledge graph first.")
                            st.stop()
                    
                    else:  # Comprehensive Analysis
                        prompt = f"""
                        You are an expert research analyst specializing in Alzheimer's disease research. 
                        Provide a comprehensive analysis of this dataset and research landscape.
                        
                        {data_summary}
                        {kg_info}
                        
                        Please provide a comprehensive analysis covering:
                        
                        **1. Data Overview**
                        - Summary of what the dataset contains
                        - Key characteristics and scope
                        
                        **2. Research Gaps Analysis**
                        - Programmatic gaps in research data
                        - Missing populations, methods, or technologies
                        - Collaboration opportunities
                        
                        **3. Data Quality Assessment**
                        - Strengths of the current data
                        - Limitations and areas for improvement
                        - FAIR compliance insights
                        
                        **4. Network Analysis** (if knowledge graph available)
                        - Collaboration patterns
                        - Research clusters
                        - Centrality insights
                        
                        **5. Strategic Recommendations**
                        - How to address identified gaps
                        - Future research priorities
                        - Process improvements
                        
                        **6. Impact Assessment**
                        - Current research impact
                        - Potential for future impact
                        
                        Be specific, actionable, and evidence-based in your recommendations.
                        """
                    
                    # Call Claude API
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0.3,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    # Display the analysis
                    st.markdown("#### 📊 AI Analysis Results")
                    st.markdown(message.content[0].text)
                    
                    # Export option
                    if st.button("Export Analysis as Text", key="export_analysis"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"ai_analysis_{analysis_type.lower().replace(' ', '_')}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"AI Analysis: {analysis_type}\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Original data: {len(studies_df)} rows\n")
                            f.write(f"Filtered data: {len(filtered_df)} rows\n\n")
                            f.write(message.content[0].text)
                        st.success(f"Analysis exported to {filename}")
                
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
                    st.info("Please check your API key and try again.")

# Tab 1: Publications
with tab1:
    _, pubmed_df, _, _ = load_data()
    
    st.markdown("""
    ### How this list was generated:
    This list was generated by searching PubMed Central for articles related to each study in our inventory.
    The search included articles published in the last 3 years that mention either the study name or abbreviation.
    For each article, we extracted the title, authors, affiliations, abstract, and keywords.
    """)
    
    # Search
    search_query = st.text_input(
        "Search publications (searches across all columns)",
        key="pubmed_search"
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        # Study name filter
        study_names = st.multiselect(
            "Filter by Study Name",
            options=sorted([str(x) for x in pubmed_df["Study Name"].dropna().unique()]),
            default=[],
            key="pubmed_study_names"
        )
        
        # Coarse Data Types filter
        coarse_types = st.multiselect(
            "Filter by Coarse Data Types",
            options=sorted(set(ct for cts in pubmed_df["Coarse Data Types"].dropna() 
                             for ct in str(cts).split(", ") if ct.strip())),
            default=[],
            key="pubmed_coarse_types"
        )
        
        # Granular Data Types filter
        granular_types = st.multiselect(
            "Filter by Granular Data Types",
            options=sorted(set(gt for gts in pubmed_df["Granular Data Types"].dropna() 
                             for gt in str(gts).split("; ") if gt.strip())),
            default=[],
            key="pubmed_granular_types"
        )
        
        # Granular Statuses filter
        granular_statuses = st.multiselect(
            "Filter by Granular Statuses",
            options=sorted(set(gs for gss in pubmed_df["Granular Statuses Included"].dropna() for gs in gss.split("; "))),
            default=[],
            key="pubmed_granular_statuses"
        )
    with col2:
        # Coarse Statuses filter
        coarse_statuses = st.multiselect(
            "Filter by Coarse Statuses",
            options=sorted(set(cs for css in pubmed_df["Coarse Statuses Included"].dropna() for cs in css.split(", ") if cs.strip())),
            default=[],
            key="pubmed_coarse_statuses"
        )
        
        # Keyword filter
        keywords = st.multiselect(
            "Filter by Keywords",
            options=sorted(set(k for kw in pubmed_df["Keywords"].dropna() for k in kw.split("; "))),
            default=[],
            key="pubmed_keywords"
        )
        
        # Authors filter
        authors = st.multiselect(
            "Filter by Authors",
            options=sorted(set(a for auth in pubmed_df["Authors"].dropna() for a in auth.split("; "))),
            default=[],
            key="pubmed_authors"
        )
    
    # Filter data
    filtered_df = pubmed_df.copy()
    if search_query:
        filtered_df = perform_text_search(filtered_df, search_query, list(filtered_df.columns))
    if study_names:
        filtered_df = filtered_df[filtered_df["Study Name"].astype(str).isin(study_names)]
    if coarse_types:
        filtered_df = filtered_df[filtered_df["Coarse Data Types"].apply(lambda x: any(ct in str(x).split(", ") for ct in coarse_types))]
    if granular_types:
        filtered_df = filtered_df[filtered_df["Granular Data Types"].apply(lambda x: any(gt in str(x).split("; ") for gt in granular_types))]
    if granular_statuses:
        filtered_df = filtered_df[filtered_df["Granular Statuses Included"].apply(lambda x: any(gs in str(x).split("; ") for gs in granular_statuses))]
    if coarse_statuses:
        filtered_df = filtered_df[filtered_df["Coarse Statuses Included"].apply(lambda x: any(cs in str(x).split(", ") for cs in coarse_statuses))]
    if keywords:
        filtered_df = filtered_df[filtered_df["Keywords"].apply(lambda x: any(k in str(x).split("; ") for k in keywords))]
    if authors:
        filtered_df = filtered_df[filtered_df["Authors"].apply(lambda x: any(a in str(x).split("; ") for a in authors))]
    
    # Display filtered data
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data matches the current filters.")

    # Knowledge Graph Buttons
    st.markdown("### Knowledge Graphs")
    st.markdown("""
    **Gold Highlights:** The top 3 most connected nodes in each graph are highlighted with gold borders to identify the most influential entities.
    
    **Color Coding:** 
    - **Authors Graph:** Orange = Highly connected, Green = Moderately connected, Blue = Less connected
    - **Affiliations Graph:** Red = Highly active, Purple = Moderately active, Brown = Less active
    """)
    
    if st.button("Show Authors Knowledge Graph", key="authors_graph"):
        st.markdown("### Authors Knowledge Graph")
        st.markdown("Interactive network visualization showing co-authorship connections between authors. Most connected authors are highlighted in gold.")
        
        # Create and display authors knowledge graph using filtered data
        if len(filtered_df) > 0:
            G = create_authors_knowledge_graph(filtered_df)
            fig = plot_knowledge_graph(G, "Authors Knowledge Graph", graph_type='authors')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Create and display summary table
            st.markdown("### 📊 Graph Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Authors", len(G.nodes()))
            with col2:
                st.metric("Total Connections", len(G.edges()))
            with col3:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Connections per Author", f"{avg_degree:.1f}")
            with col4:
                highly_connected = sum(1 for node in G.nodes() if dict(G.degree())[node] >= max(dict(G.degree()).values()) * 0.8) if len(G.nodes()) > 0 else 0
                st.metric("Highly Connected Authors", highly_connected)
            
            # Show most connected authors
            if len(G.nodes()) > 0:
                degrees = dict(G.degree())
                top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.markdown("#### 🔗 Most Connected Authors")
                for author, degree in top_connected:
                    st.write(f"• **{author}** - {degree} connections")
            
            # Detailed summary table
            summary_df = create_authors_knowledge_graph_summary(G, filtered_df)
            if summary_df is not None:
                st.markdown("#### 📋 Detailed Summary")
                st.dataframe(summary_df, use_container_width=True)
                
                # Export summary option
                if st.button("Export Authors Summary as CSV", key="authors_summary_export"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"authors_knowledge_graph_summary_{timestamp}.csv"
                    summary_df.to_csv(filename, index=False)
                    st.success(f"Authors summary exported to {filename}")
        else:
            st.warning("No data available to create authors knowledge graph.")
    
    if st.button("Show Affiliations Knowledge Graph", key="affiliations_graph"):
        st.markdown("### Affiliations Knowledge Graph")
        st.markdown("Interactive network visualization showing co-affiliation connections between institutions. Most connected affiliations are highlighted in gold.")
        
        # Create and display affiliations knowledge graph using filtered data
        if len(filtered_df) > 0:
            G = create_affiliations_knowledge_graph(filtered_df)
            fig = plot_knowledge_graph(G, "Affiliations Knowledge Graph", graph_type='affiliations')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Create and display summary table
            st.markdown("### 📊 Graph Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Affiliations", len(G.nodes()))
            with col2:
                st.metric("Total Connections", len(G.edges()))
            with col3:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Connections per Affiliation", f"{avg_degree:.1f}")
            with col4:
                highly_active = sum(1 for node in G.nodes() if dict(G.degree())[node] >= max(dict(G.degree()).values()) * 0.8) if len(G.nodes()) > 0 else 0
                st.metric("Highly Active Affiliations", highly_active)
            
            # Show most connected affiliations
            if len(G.nodes()) > 0:
                degrees = dict(G.degree())
                top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.markdown("#### 🔗 Most Connected Affiliations")
                for affiliation, degree in top_connected:
                    st.write(f"• **{affiliation}** - {degree} connections")
            
            # Detailed summary table
            summary_df = create_affiliations_knowledge_graph_summary(G, filtered_df)
            if summary_df is not None:
                st.markdown("#### 📋 Detailed Summary")
                st.dataframe(summary_df, use_container_width=True)
                
                # Export summary option
                if st.button("Export Affiliations Summary as CSV", key="affiliations_summary_export"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"affiliations_knowledge_graph_summary_{timestamp}.csv"
                    summary_df.to_csv(filename, index=False)
                    st.success(f"Affiliations summary exported to {filename}")
        else:
            st.warning("No data available to create affiliations knowledge graph.")

    # AI-Powered Analysis
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Publications Analysis")
    st.markdown("Get AI insights about publication patterns, collaboration networks, and research trends.")
    
    # Check if Anthropic API key is available
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    
    if not anthropic_api_key:
        st.info("💡 **AI Analysis Available**: Add your Anthropic API key to `.streamlit/secrets.toml` to enable AI-powered insights.")
        st.code("ANTHROPIC_API_KEY = 'your-api-key-here'")
    else:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["Publication Trends Analysis", "Collaboration Network Insights", "Research Topic Analysis", "Author/Affiliation Patterns", "Filtered vs Original Comparison", "Research Gaps vs Current Priorities", "Comprehensive Publications Analysis"],
            key="pub_analysis_type"
        )
        
        if st.button("Generate AI Analysis", key="pub_ai_analysis"):
            with st.spinner("🤖 Claude is analyzing your publications data..."):
                try:
                    # Prepare data summary for analysis
                    data_summary = f"""
                    Original publications dataset: {len(pubmed_df)} rows, {len(pubmed_df.columns)} columns
                    Filtered publications dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns
                    
                    Original columns: {list(pubmed_df.columns)}
                    Filtered columns: {list(filtered_df.columns)}
                    
                    Sample of original data (first 3 rows):
                    {pubmed_df.head(3).to_string()}
                    
                    Sample of filtered data (first 3 rows):
                    {filtered_df.head(3).to_string()}
                    """
                    
                    # Add knowledge graph info if available
                    kg_info = ""
                    if 'authors_G' in locals() and len(authors_G.nodes()) > 0:
                        authors_degrees = dict(authors_G.degree())
                        top_authors = sorted(authors_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                        kg_info += f"""
                        Authors Knowledge Graph:
                        - Total authors: {len(authors_G.nodes())}
                        - Total collaborations: {len(authors_G.edges())}
                        - Top 5 most collaborative authors: {top_authors}
                        """
                    
                    if 'affiliations_G' in locals() and len(affiliations_G.nodes()) > 0:
                        affil_degrees = dict(affiliations_G.degree())
                        top_affiliations = sorted(affil_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                        kg_info += f"""
                        Affiliations Knowledge Graph:
                        - Total affiliations: {len(affiliations_G.nodes())}
                        - Total co-affiliations: {len(affiliations_G.edges())}
                        - Top 5 most connected affiliations: {top_affiliations}
                        """
                    
                    # Create prompts based on analysis type
                    if analysis_type == "Publication Trends Analysis":
                        prompt = f"""
                        You are an expert bibliometric analyst specializing in Alzheimer's disease research publications. 
                        Analyze this publications dataset and identify key trends.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Publication Volume Trends**: How has publication volume changed over time?
                        2. **Research Focus Evolution**: How have research topics evolved?
                        3. **Geographic Distribution**: What are the geographic patterns in research output?
                        4. **Journal Analysis**: What are the most prominent journals and their characteristics?
                        5. **Citation Patterns**: What are the citation trends and impact factors?
                        6. **Emerging Topics**: What new research areas are emerging?
                        7. **Recommendations**: How could publication strategies be improved?
                        
                        Be specific and provide actionable insights for research planning.
                        """
                    
                    elif analysis_type == "Collaboration Network Insights":
                        if kg_info:
                            prompt = f"""
                            You are an expert in research collaboration networks and social network analysis. 
                            Analyze the collaboration patterns in this Alzheimer's disease research dataset.
                            
                            {data_summary}
                            {kg_info}
                            
                            Please provide:
                            1. **Network Structure Analysis**: What does the collaboration network structure reveal?
                            2. **Collaboration Patterns**: What types of collaborations are most common?
                            3. **Research Communities**: Are there distinct research communities or clusters?
                            4. **Centrality Insights**: What do the most connected authors/affiliations represent?
                            5. **Collaboration Gaps**: What collaboration opportunities are missing?
                            6. **Cross-Institutional Patterns**: How do institutions collaborate?
                            7. **Strategic Recommendations**: How could collaboration be enhanced?
                            
                            Focus on actionable insights for research coordination and networking.
                            """
                        else:
                            st.warning("No knowledge graph available. Please generate knowledge graphs first.")
                            st.stop()
                    
                    elif analysis_type == "Research Topic Analysis":
                        prompt = f"""
                        You are an expert in research topic analysis and text mining. 
                        Analyze the research topics and themes in this Alzheimer's disease publications dataset.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Topic Clustering**: What are the main research topic clusters?
                        2. **Keyword Analysis**: What are the most frequent and significant keywords?
                        3. **Topic Evolution**: How have research topics evolved over time?
                        4. **Interdisciplinary Patterns**: What interdisciplinary connections exist?
                        5. **Research Gaps**: What topics are underrepresented?
                        6. **Emerging Areas**: What new research directions are emerging?
                        7. **Strategic Recommendations**: How could topic coverage be improved?
                        
                        Use evidence from the data to support your analysis.
                        """
                    
                    elif analysis_type == "Author/Affiliation Patterns":
                        prompt = f"""
                        You are an expert in research productivity and institutional analysis. 
                        Analyze the author and affiliation patterns in this Alzheimer's disease research dataset.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Author Productivity**: What are the patterns in author productivity?
                        2. **Institutional Analysis**: What institutions are most active in research?
                        3. **Geographic Distribution**: How is research distributed geographically?
                        4. **Collaboration Patterns**: How do authors and institutions collaborate?
                        5. **Research Leadership**: Who are the key research leaders and institutions?
                        6. **Capacity Building**: What capacity building opportunities exist?
                        7. **Strategic Recommendations**: How could research capacity be enhanced?
                        
                        Focus on identifying opportunities for research development and collaboration.
                        """
                    
                    elif analysis_type == "Filtered vs Original Comparison":
                        prompt = f"""
                        You are an expert data analyst. Compare the filtered dataset to the original dataset.
                        
                        {data_summary}
                        
                        Please analyze:
                        1. **Data Reduction**: How much data was filtered out and why?
                        2. **Pattern Changes**: How do the filtered results differ from the original?
                        3. **Bias Assessment**: Are there any biases introduced by the filtering?
                        4. **Representativeness**: How representative is the filtered subset?
                        5. **Research Implications**: What does this filtering tell us about research priorities?
                        6. **Improvement Opportunities**: How could the research process be improved?
                        
                        Focus on meaningful differences and their implications.
                        """
                    
                    elif analysis_type == "Research Gaps vs Current Priorities":
                        prompt = f"""
                        You are an expert in Alzheimer's disease research and current biomedical research priorities. 
                        Analyze the research gaps in this publications dataset compared to current priority biomedical research topics.
                        
                        {data_summary}
                        
                        Please provide a comprehensive gap analysis:
                        
                        1. **Current Priority Biomedical Research Areas** (2024-2025)
                           - Early detection and biomarkers for Alzheimer's disease
                           - Precision medicine and personalized treatment approaches
                           - Novel therapeutic targets and drug development
                           - Digital health technologies and AI/ML applications
                           - Prevention strategies and lifestyle interventions
                           - Caregiver support and quality of life research
                           - Health disparities and equitable access to care
                           - Multi-omics approaches and systems biology
                           - Clinical trial design and patient recruitment
                           - Real-world evidence and implementation science
                        
                        2. **Research Coverage Analysis**
                           - How well does the current research cover these priority areas?
                           - Which priority areas are well-represented in the publications?
                           - Which priority areas are underrepresented or missing?
                           - What is the balance between basic science and clinical research?
                        
                        3. **Gap Identification** (if filtered data exists, compare both)
                           - What specific research gaps exist in the original dataset?
                           - How do the gaps change when comparing filtered vs. original data?
                           - Which priority areas are most affected by the filtering?
                           - What research questions remain unanswered?
                        
                        4. **Emerging vs. Established Research Areas**
                           - How does the research balance emerging vs. established topics?
                           - Are there opportunities for cross-disciplinary research?
                           - What innovative approaches are being explored?
                           - What traditional approaches might need updating?
                        
                        5. **Geographic and Institutional Gaps**
                           - Are certain priority areas concentrated in specific regions?
                           - Which institutions are leading in priority research areas?
                           - What geographic gaps exist in priority research?
                           - How could collaboration address these gaps?
                        
                        6. **Strategic Recommendations**
                           - How should research priorities be adjusted to address gaps?
                           - What funding opportunities should be pursued?
                           - How could collaboration be enhanced to address gaps?
                           - What capacity building is needed?
                           - How could the research community better align with priorities?
                        
                        7. **Future Research Directions**
                           - What emerging research areas should be prioritized?
                           - How could technology and innovation address gaps?
                           - What partnerships could accelerate progress?
                           - How could patient and caregiver perspectives inform priorities?
                        
                        Be specific, evidence-based, and provide actionable recommendations for addressing research gaps.
                        """
                    
                    else:  # Comprehensive Publications Analysis
                        prompt = f"""
                        You are an expert research analyst specializing in Alzheimer's disease research publications. 
                        Provide a comprehensive analysis of this publications dataset and research landscape.
                        
                        {data_summary}
                        {kg_info}
                        
                        Please provide a comprehensive analysis covering:
                        
                        **1. Publications Overview**
                        - Summary of publication volume and scope
                        - Key characteristics and trends
                        
                        **2. Research Topic Analysis**
                        - Main research themes and clusters
                        - Topic evolution over time
                        - Emerging research areas
                        
                        **3. Collaboration Network Analysis**
                        - Author collaboration patterns
                        - Institutional collaboration networks
                        - Research community structure
                        
                        **4. Research Impact Assessment**
                        - Citation patterns and impact
                        - High-impact publications
                        - Research influence analysis
                        
                        **5. Geographic and Institutional Analysis**
                        - Geographic distribution of research
                        - Leading institutions and researchers
                        - Capacity building opportunities
                        
                        **6. Strategic Recommendations**
                        - How to enhance research collaboration
                        - Future research priorities
                        - Publication strategy improvements
                        
                        **7. Research Gaps and Opportunities**
                        - Underrepresented topics
                        - Collaboration opportunities
                        - Capacity development needs
                        
                        Be specific, actionable, and evidence-based in your recommendations.
                        """
                    
                    # Call Claude API
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0.3,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    # Display the analysis
                    st.markdown("#### 📊 AI Analysis Results")
                    st.markdown(message.content[0].text)
                    
                    # Export option
                    if st.button("Export Analysis as Text", key="export_pub_analysis"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"publications_ai_analysis_{analysis_type.lower().replace(' ', '_')}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Publications AI Analysis: {analysis_type}\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Original data: {len(pubmed_df)} rows\n")
                            f.write(f"Filtered data: {len(filtered_df)} rows\n\n")
                            f.write(message.content[0].text)
                        st.success(f"Analysis exported to {filename}")
                
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
                    st.info("Please check your API key and try again.")

# Tab 2: Code
with tab2:
    _, _, github_df, _ = load_data()
    
    # Pre-filter to show only biomedical repositories (YES - in Biomedical Relevance)
    if 'Biomedical Relevance' in github_df.columns:
        # Filter for repositories that start with "YES -" in Biomedical Relevance
        biomedical_mask = github_df['Biomedical Relevance'].astype(str).str.startswith('YES -', na=False)
        github_df = github_df[biomedical_mask].copy()
    
    st.markdown("""
    ### How this list was generated:
    This list was generated by searching GitHub for repositories related to each study in our inventory.
    The search included repositories that mention both the study abbreviation and "alzheimer".
    For each repository, we extracted the owner, contributors, languages, and analyzed the README/content
    to identify data types, modalities, and tools used.
    
    **Note:** This view shows only repositories that have been identified as biomedical research related (marked with "YES -" in Biomedical Relevance).
    """)
    
    # Search
    search_query = st.text_input(
        "Search code repositories (searches across all columns)",
        key="github_search"
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        # Study name filter
        study_names = st.multiselect(
            "Filter by Study Name",
            options=sorted([str(x) for x in github_df["Study Name"].dropna().unique()]),
            default=[],
            key="github_study_names"
        )
        
        # Abbreviation filter
        abbreviations = st.multiselect(
            "Filter by Abbreviation",
            options=sorted([str(x) for x in github_df["Abbreviation"].dropna().unique()]),
            default=[],
            key="github_abbreviations"
        )
        
        # Diseases Included filter
        diseases = st.multiselect(
            "Filter by Diseases Included",
            options=sorted(set(d for diseases in github_df["Diseases Included"].dropna() 
                             for d in str(diseases).split("; ") if d.strip())),
            default=[],
            key="github_diseases"
        )
        
        # Languages filter
        unique_languages = sorted([str(x) for x in github_df["Languages"].dropna().unique() if str(x).strip()])
        languages = st.multiselect(
            "Filter by Languages",
            options=unique_languages,
            default=[],
            key="github_languages"
        )
    with col2:
        # Data Types filter
        data_types_text = []
        for data_types_str in github_df["Data Types"].dropna():
            if str(data_types_str).strip():
                # Split by common delimiters and clean up
                data_parts = str(data_types_str).replace(" - ", "; ").replace(" - ", "; ").split("; ")
                data_types_text.extend([d.strip() for d in data_parts if d.strip()])
        unique_data_types = sorted(list(set(data_types_text)))
        data_types = st.multiselect(
            "Filter by Data Types",
            options=unique_data_types,
            default=[],
            key="github_data_types"
        )
        
        # Tooling filter
        tools_text = []
        for tools_str in github_df["Tooling"].dropna():
            if str(tools_str).strip():
                # Split by common delimiters and clean up
                tools_parts = str(tools_str).replace(" - ", "; ").replace(" - ", "; ").split("; ")
                tools_text.extend([t.strip() for t in tools_parts if t.strip()])
        unique_tools = sorted(list(set(tools_text)))
        tools = st.multiselect(
            "Filter by Tooling",
            options=unique_tools,
            default=[],
            key="github_tools"
        )
    
    # Filter data
    filtered_df = github_df.copy()
    if search_query:
        filtered_df = perform_text_search(filtered_df, search_query, list(filtered_df.columns))
    if study_names:
        filtered_df = filtered_df[filtered_df["Study Name"].astype(str).isin(study_names)]
    if abbreviations:
        filtered_df = filtered_df[filtered_df["Abbreviation"].astype(str).isin(abbreviations)]
    if diseases:
        filtered_df = filtered_df[filtered_df["Diseases Included"].apply(lambda x: any(d in str(x).split("; ") for d in diseases))]
    if languages:
        filtered_df = filtered_df[filtered_df["Languages"].astype(str).isin(languages)]
    if data_types:
        # For data types, check if any of the selected data types appear in the Data Types text
        def contains_data_type(data_types_text, selected_data_types):
            if pd.isna(data_types_text) or not str(data_types_text).strip():
                return False
            data_types_str = str(data_types_text).lower()
            return any(data_type.lower() in data_types_str for data_type in selected_data_types)
        
        filtered_df = filtered_df[filtered_df["Data Types"].apply(lambda x: contains_data_type(x, data_types))]
    if tools:
        # For tools, check if any of the selected tools appear in the Tooling text
        def contains_tool(tools_text, selected_tools):
            if pd.isna(tools_text) or not str(tools_text).strip():
                return False
            tools_str = str(tools_text).lower()
            return any(tool.lower() in tools_str for tool in selected_tools)
        
        filtered_df = filtered_df[filtered_df["Tooling"].apply(lambda x: contains_tool(x, tools))]
    
    # Display filtered data
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data matches the current filters.")

    # Export filtered data
    if st.button("Export Filtered Data as CSV", key="export_filtered_github"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"github_repositories_filtered_{timestamp}.csv"
        filtered_df.to_csv(filename, index=False)
        st.success(f"Filtered data exported to {filename}")

    # Knowledge Graph for Code Repositories
    st.markdown("### 🔗 Knowledge Graph: Repository Connections")
    st.markdown("""
    This graph shows connections between repositories based on shared content in Biomedical Relevance, Code Summary, Data Types, and Tooling columns.
    Nodes represent repositories, colored by study abbreviation. Hover to see owner and repository link.
    """)
    
    if st.button("Generate Knowledge Graph", key="code_graph"):
        with st.spinner("Generating knowledge graph..."):
            # Create graph from the five rightmost columns using filtered data
            G = nx.Graph()
            
            # Add nodes for each repository from filtered data
            for idx, row in filtered_df.iterrows():
                repo_name = f"{row['Owner']}/{row['Repository Link'].split('/')[-1]}"
                G.add_node(repo_name, 
                          abbreviation=row['Abbreviation'],
                          owner=row['Owner'],
                          repo_link=row['Repository Link'])
            
            # Create connections based on shared content in the five rightmost columns
            rightmost_cols = ['Biomedical Relevance', 'Code Summary', 'Data Types', 'Tooling']
            
            for i, row1 in filtered_df.iterrows():
                for j, row2 in filtered_df.iterrows():
                    if i >= j:  # Avoid duplicate connections
                        continue
                    
                    # Check for shared content in each column
                    shared_content = 0
                    for col in rightmost_cols:
                        if pd.notna(row1[col]) and pd.notna(row2[col]):
                            # Convert to string and find common words
                            text1 = str(row1[col]).lower()
                            text2 = str(row2[col]).lower()
                            
                            # Split into words and find common ones
                            words1 = set(text1.split())
                            words2 = set(text2.split())
                            common_words = words1.intersection(words2)
                            
                            # If they share meaningful content (more than 2 common words)
                            if len(common_words) > 2:
                                shared_content += 1
                    
                    # Add edge if they share content in at least 2 columns
                    if shared_content >= 2:
                        repo1 = f"{row1['Owner']}/{row1['Repository Link'].split('/')[-1]}"
                        repo2 = f"{row2['Owner']}/{row2['Repository Link'].split('/')[-1]}"
                        G.add_edge(repo1, repo2, weight=shared_content)

            if len(G.nodes()) > 0:
                # Create color mapping for abbreviations - handle case sensitivity
                abbreviations = list(set([G.nodes[node]['abbreviation'] for node in G.nodes()]))
                colors = px.colors.qualitative.Set3[:len(abbreviations)]
                color_map = dict(zip(abbreviations, colors))
                
                # Create node colors with error handling for missing abbreviations
                node_colors = []
                for node in G.nodes():
                    abbreviation = G.nodes[node]['abbreviation']
                    if abbreviation in color_map:
                        node_colors.append(color_map[abbreviation])
                    else:
                        # Fallback color for any missing abbreviations
                        node_colors.append('#808080')  # Gray
                
                # Create hover text
                hover_text = []
                for node in G.nodes():
                    owner = G.nodes[node]['owner']
                    repo_link = G.nodes[node]['repo_link']
                    abbreviation = G.nodes[node]['abbreviation']
                    hover_text.append(f"<b>{node}</b><br>Study: {abbreviation}<br>Owner: {owner}<br>Link: {repo_link}")
                
                # Create the plot
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                fig = go.Figure()
                
                # Add edges
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'))
                
                # Add nodes
                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    hovertext=hover_text,
                    marker=dict(
                        color=node_colors,
                        size=10,
                        line_width=2,
                        line_color='white'
                    ),
                    text=[node for node in G.nodes()],
                    textposition="middle center",
                    textfont=dict(size=8, color='white')
                ))
                
                fig.update_layout(
                    title='Repository Connections Based on Shared Content',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Nodes colored by study abbreviation<br>Hover for repository details",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=10)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📊 Graph Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Repositories", len(G.nodes()))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                    st.metric("Avg Connections per Repo", f"{avg_degree:.1f}")
                with col4:
                    studies_represented = len(set([G.nodes[node]['abbreviation'] for node in G.nodes()]))
                    st.metric("Studies Represented", studies_represented)
                
                # Show most connected repositories
                if len(G.nodes()) > 0:
                    degrees = dict(G.degree())
                    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    st.markdown("#### 🔗 Most Connected Repositories")
                    for repo, degree in top_connected:
                        abbreviation = G.nodes[repo]['abbreviation']
                        owner = G.nodes[repo]['owner']
                        st.write(f"• **{repo}** ({abbreviation}) - {degree} connections - Owner: {owner}")
                
                # Create and display detailed summary table
                st.markdown("#### 📋 Detailed Summary")
                
                # Create summary dataframe
                summary_data = []
                for node in G.nodes():
                    degree = dict(G.degree())[node]
                    abbreviation = G.nodes[node]['abbreviation']
                    owner = G.nodes[node]['owner']
                    repo_link = G.nodes[node]['repo_link']
                    
                    # Get connected repositories
                    neighbors = list(G.neighbors(node))
                    connected_repos = "; ".join(neighbors[:5])  # Show first 5 connections
                    if len(neighbors) > 5:
                        connected_repos += f" (+{len(neighbors)-5} more)"
                    
                    summary_data.append({
                        'Repository': node,
                        'Study': abbreviation,
                        'Owner': owner,
                        'Repository Link': repo_link,
                        'Connections': degree,
                        'Connected Repositories': connected_repos
                    })
                
                # Sort by number of connections
                summary_df = pd.DataFrame(summary_data).sort_values('Connections', ascending=False)
                st.dataframe(summary_df, use_container_width=True)
                
                # Export summary option
                if st.button("Export Summary as CSV", key="code_summary_export"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"code_repositories_knowledge_graph_summary_{timestamp}.csv"
                    summary_df.to_csv(filename, index=False)
                    st.success(f"Summary exported to {filename}")
            else:
                st.warning("No connections found between repositories based on the specified criteria.")

    # AI-Powered Analysis
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Code Repository Analysis")
    st.markdown("Get AI insights about code repositories, development patterns, and research software trends.")
    
    # Check if Anthropic API key is available
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    
    if not anthropic_api_key:
        st.info("💡 **AI Analysis Available**: Add your Anthropic API key to `.streamlit/secrets.toml` to enable AI-powered insights.")
        st.code("ANTHROPIC_API_KEY = 'your-api-key-here'")
    else:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["Repository Technology Analysis", "Development Patterns", "Research Software Trends", "Collaboration Insights", "Comprehensive Code Analysis"],
            key="code_analysis_type"
        )
        
        if st.button("Generate AI Analysis", key="code_ai_analysis"):
            with st.spinner("🤖 Claude is analyzing your code repositories..."):
                try:
                    # Prepare data summary for analysis
                    data_summary = f"""
                    Original code repositories dataset: {len(github_df)} rows, {len(github_df.columns)} columns
                    Filtered code repositories dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns
                    
                    Original columns: {list(github_df.columns)}
                    Filtered columns: {list(filtered_df.columns)}
                    
                    Sample of original data (first 3 rows):
                    {github_df.head(3).to_string()}
                    
                    Sample of filtered data (first 3 rows):
                    {filtered_df.head(3).to_string()}
                    """
                    
                    # Add knowledge graph info if available
                    kg_info = ""
                    if 'G' in locals() and len(G.nodes()) > 0:
                        degrees = dict(G.degree())
                        top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                        kg_info = f"""
                        Repository Knowledge Graph:
                        - Total repositories: {len(G.nodes())}
                        - Total connections: {len(G.edges())}
                        - Top 5 most connected repositories: {top_connected}
                        """
                    
                    # Create prompts based on analysis type
                    if analysis_type == "Repository Technology Analysis":
                        prompt = f"""
                        You are an expert in research software engineering and technology analysis. 
                        Analyze this code repositories dataset to understand technology patterns in Alzheimer's disease research.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Programming Language Trends**: What languages are most commonly used and why?
                        2. **Technology Stack Analysis**: What technology stacks and frameworks are prevalent?
                        3. **Data Processing Tools**: What tools are used for data analysis and processing?
                        4. **Research Software Patterns**: What patterns emerge in research software development?
                        5. **Technology Gaps**: What technologies or tools are missing?
                        6. **Best Practices**: What development practices are evident?
                        7. **Recommendations**: How could technology adoption be improved?
                        
                        Focus on actionable insights for research software development.
                        """
                    
                    elif analysis_type == "Development Patterns":
                        prompt = f"""
                        You are an expert in software development patterns and research software engineering. 
                        Analyze the development patterns in this Alzheimer's disease research code repositories.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Repository Structure Patterns**: What are common repository structures?
                        2. **Development Workflow Analysis**: What development workflows are evident?
                        3. **Code Organization**: How is code typically organized?
                        4. **Documentation Patterns**: What documentation practices are used?
                        5. **Collaboration Models**: How do developers collaborate on research software?
                        6. **Quality Assurance**: What quality assurance practices are evident?
                        7. **Recommendations**: How could development practices be improved?
                        
                        Focus on identifying best practices and improvement opportunities.
                        """
                    
                    elif analysis_type == "Research Software Trends":
                        prompt = f"""
                        You are an expert in research software trends and scientific computing. 
                        Analyze the trends in research software development for Alzheimer's disease research.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Software Categories**: What types of research software are most common?
                        2. **Tool Evolution**: How have tools and technologies evolved?
                        3. **Research Software Ecosystem**: What does the ecosystem look like?
                        4. **Integration Patterns**: How do different tools integrate?
                        5. **Emerging Technologies**: What new technologies are being adopted?
                        6. **Sustainability**: What sustainability patterns are evident?
                        7. **Future Trends**: What trends are likely to emerge?
                        
                        Use evidence from the data to support your analysis.
                        """
                    
                    elif analysis_type == "Collaboration Insights":
                        if kg_info:
                            prompt = f"""
                            You are an expert in research collaboration and software development networks. 
                            Analyze the collaboration patterns in this Alzheimer's disease research software ecosystem.
                            
                            {data_summary}
                            {kg_info}
                            
                            Please provide:
                            1. **Collaboration Network Analysis**: What does the repository network reveal?
                            2. **Development Communities**: Are there distinct development communities?
                            3. **Cross-Repository Patterns**: How do repositories relate to each other?
                            4. **Technology Sharing**: How is technology shared between projects?
                            5. **Collaboration Gaps**: What collaboration opportunities are missing?
                            6. **Knowledge Transfer**: How is knowledge transferred between projects?
                            7. **Strategic Recommendations**: How could collaboration be enhanced?
                            
                            Focus on actionable insights for research software collaboration.
                            """
                        else:
                            st.warning("No knowledge graph available. Please generate knowledge graph first.")
                            st.stop()
                    
                    else:  # Comprehensive Code Analysis
                        prompt = f"""
                        You are an expert research software analyst specializing in Alzheimer's disease research. 
                        Provide a comprehensive analysis of this code repositories dataset and research software landscape.
                        
                        {data_summary}
                        {kg_info}
                        
                        Please provide a comprehensive analysis covering:
                        
                        **1. Repository Overview**
                        - Summary of repository types and characteristics
                        - Key technology patterns and trends
                        
                        **2. Technology Stack Analysis**
                        - Programming languages and frameworks
                        - Data processing and analysis tools
                        - Research software patterns
                        
                        **3. Development Practices Analysis**
                        - Repository organization patterns
                        - Documentation and quality practices
                        - Collaboration workflows
                        
                        **4. Research Software Ecosystem**
                        - Software categories and purposes
                        - Integration patterns and dependencies
                        - Sustainability considerations
                        
                        **5. Collaboration Network Analysis** (if knowledge graph available)
                        - Repository relationships
                        - Development communities
                        - Knowledge sharing patterns
                        
                        **6. Strategic Recommendations**
                        - How to improve research software development
                        - Technology adoption strategies
                        - Collaboration enhancement opportunities
                        
                        **7. Future Directions**
                        - Emerging technology trends
                        - Research software priorities
                        - Capacity building needs
                        
                        Be specific, actionable, and evidence-based in your recommendations.
                        """
                    
                    # Call Claude API
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0.3,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    # Display the analysis
                    st.markdown("#### 📊 AI Analysis Results")
                    st.markdown(message.content[0].text)
                    
                    # Export option
                    if st.button("Export Analysis as Text", key="export_code_analysis"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"code_repositories_ai_analysis_{analysis_type.lower().replace(' ', '_')}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Code Repositories AI Analysis: {analysis_type}\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Original data: {len(github_df)} rows\n")
                            f.write(f"Filtered data: {len(filtered_df)} rows\n\n")
                            f.write(message.content[0].text)
                        st.success(f"Analysis exported to {filename}")
                
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
                    st.info("Please check your API key and try again.")

# Tab 3: Biorepositories (iNDI)
with tab3:
    _, _, _, indi_df = load_data()
    
    st.markdown("""
    ### iNDI (Induced Pluripotent Stem Cell Neurodegenerative Disease Initiative)
    This biorepository contains induced pluripotent stem cell (iPSC) lines with genetic variants associated with neurodegenerative diseases.
    Each line includes detailed information about the gene, variant, genotype, and procurement details.
    """)
    
    # Search
    search_query = st.text_input(
        "Search genes and variants (searches across all columns)",
        key="indi_search"
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        # Gene filter
        genes = st.multiselect(
            "Filter by Gene",
            options=sorted([str(x) for x in indi_df["Gene"].dropna().unique()]),
            default=[],
            key="indi_genes"
        )
        
        # Granular Statuses filter
        granular_statuses = st.multiselect(
            "Filter by Granular Statuses",
            options=sorted(set(gs for gss in indi_df["Granular Statuses Included"].dropna() for gs in gss.split("; "))),
            default=[],
            key="indi_granular_statuses"
        )
        
        # Gene Variant filter
        gene_variants = st.multiselect(
            "Filter by Gene Variant",
            options=sorted([str(x) for x in indi_df["Gene Variant"].dropna().unique()]),
            default=[],
            key="indi_gene_variants"
        )
    with col2:
        # Coarse Statuses filter
        coarse_statuses = st.multiselect(
            "Filter by Coarse Statuses",
            options=sorted(set(cs for css in indi_df["Coarse Statuses Included"].dropna() for cs in css.split(", ") if cs.strip())),
            default=[],
            key="indi_coarse_statuses"
        )
        
        # dbSNP filter
        dbsnp_ids = st.multiselect(
            "Filter by dbSNP",
            options=sorted([str(x) for x in indi_df["dbSNP"].dropna().unique()]),
            default=[],
            key="indi_dbsnp"
        )
        
        # Genome Assembly filter
        assemblies = st.multiselect(
            "Filter by Genome Assembly",
            options=sorted([str(x) for x in indi_df["Genome Assembly"].dropna().unique()]),
            default=[],
            key="indi_assemblies"
        )
    
    # Filter data
    filtered_df = indi_df.copy()
    if search_query:
        filtered_df = perform_text_search(filtered_df, search_query, list(filtered_df.columns))
    if genes:
        filtered_df = filtered_df[filtered_df["Gene"].astype(str).isin(genes)]
    if granular_statuses:
        filtered_df = filtered_df[filtered_df["Granular Statuses Included"].apply(lambda x: any(gs in str(x).split("; ") for gs in granular_statuses))]
    if coarse_statuses:
        filtered_df = filtered_df[filtered_df["Coarse Statuses Included"].apply(lambda x: any(cs in str(x).split(", ") for cs in coarse_statuses))]
    if gene_variants:
        filtered_df = filtered_df[filtered_df["Gene Variant"].astype(str).isin(gene_variants)]
    if dbsnp_ids:
        filtered_df = filtered_df[filtered_df["dbSNP"].astype(str).isin(dbsnp_ids)]
    if assemblies:
        filtered_df = filtered_df[filtered_df["Genome Assembly"].astype(str).isin(assemblies)]
    
    # Display filtered data
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data matches the current filters.")

    # AI-Powered Analysis
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Biorepository Analysis")
    st.markdown("Get AI insights about gene variants, disease associations, and biorepository patterns.")
    
    # Check if Anthropic API key is available
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    
    if not anthropic_api_key:
        st.info("💡 **AI Analysis Available**: Add your Anthropic API key to `.streamlit/secrets.toml` to enable AI-powered insights.")
        st.code("ANTHROPIC_API_KEY = 'your-api-key-here'")
    else:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["Gene Variant Analysis", "Disease Association Patterns", "Biorepository Coverage", "Genetic Diversity Analysis", "Filtered vs Original Comparison", "Research Gaps in Neurodegeneration", "Comprehensive Biorepository Analysis"],
            key="biorepo_analysis_type"
        )
        
        if st.button("Generate AI Analysis", key="biorepo_ai_analysis"):
            with st.spinner("🤖 Claude is analyzing your biorepository data..."):
                try:
                    # Prepare data summary for analysis
                    data_summary = f"""
                    Original biorepository dataset: {len(indi_df)} rows, {len(indi_df.columns)} columns
                    Filtered biorepository dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns
                    
                    Original columns: {list(indi_df.columns)}
                    Filtered columns: {list(filtered_df.columns)}
                    
                    Sample of original data (first 3 rows):
                    {indi_df.head(3).to_string()}
                    
                    Sample of filtered data (first 3 rows):
                    {filtered_df.head(3).to_string()}
                    """
                    
                    # Create prompts based on analysis type
                    if analysis_type == "Gene Variant Analysis":
                        prompt = f"""
                        You are an expert geneticist specializing in neurodegenerative diseases and gene variants. 
                        Analyze this biorepository dataset to understand gene variant patterns in Alzheimer's disease research.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Gene Distribution**: What genes are most commonly represented and why?
                        2. **Variant Types**: What types of genetic variants are most prevalent?
                        3. **Functional Impact**: What are the functional implications of these variants?
                        4. **Variant Frequency**: How do variant frequencies compare across populations?
                        5. **Pathogenicity**: What are the pathogenicity patterns of these variants?
                        6. **Research Gaps**: What gene variants are missing from the repository?
                        7. **Recommendations**: How could gene variant coverage be improved?
                        
                        Focus on actionable insights for genetic research and biorepository development.
                        """
                    
                    elif analysis_type == "Disease Association Patterns":
                        prompt = f"""
                        You are an expert in disease genetics and neurodegenerative disease research. 
                        Analyze the disease association patterns in this Alzheimer's disease biorepository.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Disease-Gene Relationships**: What are the key disease-gene associations?
                        2. **Phenotype Patterns**: What phenotypic patterns emerge from the data?
                        3. **Disease Mechanisms**: What disease mechanisms are represented?
                        4. **Risk Factors**: What genetic risk factors are most common?
                        5. **Disease Progression**: How do variants relate to disease progression?
                        6. **Therapeutic Targets**: What therapeutic targets are evident?
                        7. **Research Priorities**: What should be the research priorities?
                        
                        Focus on understanding disease mechanisms and therapeutic opportunities.
                        """
                    
                    elif analysis_type == "Biorepository Coverage":
                        prompt = f"""
                        You are an expert in biorepository management and genetic resource development. 
                        Analyze the coverage and completeness of this Alzheimer's disease biorepository.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Geographic Coverage**: How is the repository distributed geographically?
                        2. **Population Diversity**: What population diversity is represented?
                        3. **Disease Spectrum**: How comprehensive is the disease spectrum coverage?
                        4. **Quality Assessment**: What is the quality of the genetic data?
                        5. **Metadata Completeness**: How complete is the metadata?
                        6. **Accessibility**: How accessible are these resources?
                        7. **Gap Analysis**: What gaps exist in the repository?
                        
                        Focus on identifying opportunities for biorepository enhancement.
                        """
                    
                    elif analysis_type == "Genetic Diversity Analysis":
                        prompt = f"""
                        You are an expert in population genetics and genetic diversity analysis. 
                        Analyze the genetic diversity patterns in this Alzheimer's disease biorepository.
                        
                        {data_summary}
                        
                        Please provide:
                        1. **Population Representation**: How diverse is the population representation?
                        2. **Genetic Variability**: What patterns of genetic variability exist?
                        3. **Allele Frequency**: What are the allele frequency patterns?
                        4. **Population Structure**: What population structure is evident?
                        5. **Admixture Patterns**: What admixture patterns are present?
                        6. **Selection Pressures**: What selection pressures are evident?
                        7. **Diversity Gaps**: What diversity gaps need to be addressed?
                        
                        Focus on understanding genetic diversity and its implications for research.
                        """
                    
                    elif analysis_type == "Filtered vs Original Comparison":
                        prompt = f"""
                        You are an expert geneticist and biomedical researcher specializing in neurodegenerative diseases and functional genomics. 
                        Compare the filtered biorepository dataset to the original biorepository dataset with a focus on genes of biomedical importance.
                        
                        {data_summary}
                        
                        Please provide a comprehensive comparison analysis:
                        
                        1. **Data Reduction Analysis**
                           - How many gene variants were filtered out (percentage and absolute numbers)?
                           - What specific filters caused the most data reduction?
                           - Which genes, variants, or disease statuses were most commonly filtered?
                           - What is the impact on rare vs. common variants?
                        
                        2. **Genes of Biomedical Importance**
                           - **High-Impact Genes**: Which genes in the original dataset have the highest potential for biomedical discovery?
                           - **Therapeutic Targets**: Which genes represent the most promising therapeutic targets?
                           - **Biomarker Candidates**: Which genes could serve as biomarkers for early detection or disease progression?
                           - **Pathway Enrichment**: Which biological pathways are most affected by the filtering?
                           - **Drug Development Potential**: Which genes have the highest potential for drug development?
                        
                        3. **Functional Validation Opportunities**
                           - **iPSC Model Potential**: Which genes would be most valuable for iPSC-based functional studies?
                           - **Mechanism Studies**: Which genes offer the best opportunities for mechanistic studies?
                           - **Phenotype-Genotype Correlations**: Which genes have the strongest phenotype-genotype relationships?
                           - **Cross-Species Validation**: Which genes have the best potential for cross-species validation studies?
                           - **Clinical Translation**: Which genes have the highest potential for clinical translation?
                        
                        4. **Content Pattern Changes**
                           - How do the gene distributions differ between original and filtered datasets?
                           - What changes occur in variant type patterns (missense, nonsense, frameshift, etc.)?
                           - How do disease associations change between datasets?
                           - What changes are evident in population representation?
                           - How does the balance of pathogenic vs. benign variants change?
                        
                        5. **Biomedical Discovery Potential**
                           - **Novel Gene Discovery**: Which genes in the original dataset represent novel discoveries?
                           - **Known Disease Genes**: How does filtering affect representation of known disease genes?
                           - **Emerging Targets**: Which genes represent emerging therapeutic targets?
                           - **Risk Factor Genes**: Which genes have the strongest association with disease risk?
                           - **Protective Variants**: Which genes might contain protective variants?
                        
                        6. **Bias and Representativeness Assessment**
                           - Are there any systematic biases introduced by the filtering?
                           - How representative is the filtered subset of the broader genetic landscape?
                           - What types of genetic variants might be underrepresented in the filtered results?
                           - Are certain genes or disease mechanisms disproportionately affected?
                           - How does filtering affect rare disease gene representation?
                        
                        7. **Research Priority Implications**
                           - What does the filtering reveal about current genetic research priorities?
                           - Which genetic pathways are being emphasized or de-emphasized?
                           - How do the filtered results align with neurodegenerative disease research priorities?
                           - Which genes should be prioritized for functional studies?
                        
                        8. **Quality and Clinical Relevance Assessment**
                           - How does the clinical relevance of variants change between datasets?
                           - Are high-impact variants more or less likely to be filtered?
                           - What does this tell us about research vs. clinical utility?
                           - Which variants have the highest clinical significance?
                        
                        9. **Strategic Recommendations for Biomedical Discovery**
                           - **Priority Genes for Study**: Which specific genes should be prioritized for functional validation?
                           - **Experimental Approaches**: What experimental approaches would be most valuable for these genes?
                           - **Collaboration Opportunities**: How could collaboration enhance functional studies?
                           - **Resource Allocation**: How should resources be allocated for gene functional studies?
                           - **Technology Needs**: What new technologies are needed for functional validation?
                        
                        10. **Future Directions**
                            - **Emerging Technologies**: How could new technologies (CRISPR, single-cell, etc.) be applied to these genes?
                            - **Multi-Omics Integration**: How could multi-omics approaches enhance understanding of these genes?
                            - **Clinical Trials**: Which genes have the highest potential for clinical trial development?
                            - **Precision Medicine**: How could these genes inform precision medicine approaches?
                        
                        Provide specific examples from the data to support your analysis, with particular attention to genes that could drive biomedical discovery and serve as functional validation studies.
                        """
                    
                    elif analysis_type == "Research Gaps in Neurodegeneration":
                        prompt = f"""
                        You are an expert in neurodegenerative disease research and genetic medicine. 
                        Analyze the research gaps in this biorepository dataset compared to current priority areas in neurodegeneration research.
                        
                        {data_summary}
                        
                        Please provide a comprehensive gap analysis:
                        
                        1. **Current Priority Neurodegeneration Research Areas** (2024-2025)
                           - Early genetic risk identification and prediction models
                           - Rare genetic variants and their functional characterization
                           - Polygenic risk scores and complex genetic interactions
                           - Gene-environment interactions and epigenetics
                           - Therapeutic target validation and drug development
                           - Precision medicine approaches for genetic subgroups
                           - Biomarker development for genetic risk stratification
                           - Clinical trial design for genetic interventions
                           - Population-specific genetic risk factors
                           - Cross-disease genetic mechanisms and comorbidities
                        
                        2. **Research Coverage Analysis**
                           - How well does the current biorepository cover these priority areas?
                           - Which priority areas are well-represented in the genetic variants?
                           - Which priority areas are underrepresented or missing?
                           - What is the balance between common and rare variants?
                        
                        3. **Gap Identification** (if filtered data exists, compare both)
                           - What specific genetic research gaps exist in the original dataset?
                           - How do the gaps change when comparing filtered vs. original data?
                           - Which priority areas are most affected by the filtering?
                           - What genetic questions remain unanswered?
                        
                        4. **Emerging vs. Established Genetic Research**
                           - How does the research balance emerging vs. established genetic pathways?
                           - Are there opportunities for cross-disease genetic research?
                           - What innovative genetic approaches are being explored?
                           - What traditional genetic approaches might need updating?
                        
                        5. **Geographic and Population Gaps**
                           - Are certain genetic variants concentrated in specific populations?
                           - Which populations are underrepresented in the biorepository?
                           - What geographic gaps exist in genetic research?
                           - How could population diversity be improved?
                        
                        6. **Strategic Recommendations**
                           - How should genetic research priorities be adjusted to address gaps?
                           - What genetic resources should be prioritized for development?
                           - How could collaboration be enhanced to address genetic gaps?
                           - What capacity building is needed for genetic research?
                           - How could the genetic research community better align with priorities?
                        
                        7. **Future Genetic Research Directions**
                           - What emerging genetic research areas should be prioritized?
                           - How could new genetic technologies address gaps?
                           - What partnerships could accelerate genetic research progress?
                           - How could patient genetic data inform research priorities?
                        
                        Be specific, evidence-based, and provide actionable recommendations for addressing genetic research gaps in neurodegeneration.
                        """
                    
                    else:  # Comprehensive Biorepository Analysis
                        prompt = f"""
                        You are an expert geneticist and biorepository analyst specializing in Alzheimer's disease research. 
                        Provide a comprehensive analysis of this biorepository dataset and genetic research landscape.
                        
                        {data_summary}
                        
                        Please provide a comprehensive analysis covering:
                        
                        **1. Biorepository Overview**
                        - Summary of gene variants and their characteristics
                        - Key genetic patterns and trends
                        
                        **2. Gene Variant Analysis**
                        - Gene distribution and representation
                        - Variant types and functional impact
                        - Pathogenicity and clinical significance
                        
                        **3. Disease Association Analysis**
                        - Disease-gene relationships
                        - Phenotypic patterns and mechanisms
                        - Therapeutic target identification
                        
                        **4. Population and Diversity Analysis**
                        - Population representation and diversity
                        - Genetic variability patterns
                        - Geographic and ethnic distribution
                        
                        **5. Quality and Coverage Assessment**
                        - Data quality and completeness
                        - Metadata standards and accessibility
                        - Repository management practices
                        
                        **6. Strategic Recommendations**
                        - How to enhance biorepository coverage
                        - Genetic research priorities
                        - Capacity building opportunities
                        
                        **7. Future Directions**
                        - Emerging genetic research areas
                        - Biorepository development priorities
                        - Collaborative opportunities
                        
                        Be specific, actionable, and evidence-based in your recommendations.
                        """
                    
                    # Call Claude API
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0.3,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    # Display the analysis
                    st.markdown("#### 📊 AI Analysis Results")
                    st.markdown(message.content[0].text)
                    
                    # Export option
                    if st.button("Export Analysis as Text", key="export_biorepo_analysis"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"biorepository_ai_analysis_{analysis_type.lower().replace(' ', '_')}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Biorepository AI Analysis: {analysis_type}\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Original data: {len(indi_df)} rows\n")
                            f.write(f"Filtered data: {len(filtered_df)} rows\n\n")
                            f.write(message.content[0].text)
                        st.success(f"Analysis exported to {filename}")
                
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
                    st.info("Please check your API key and try again.")

# Tab 4: About
with tab4:
    st.markdown("""
    # CARD Catalogue

    A FAIR browser for publicly available and controlled access Alzheimer's disease studies with AI-powered insights.

    ## Overview

    The CARD Catalogue is a comprehensive web application that provides an advanced view of Alzheimer's disease studies and their associated resources. It features four main sections with AI-powered analysis capabilities:

    1. **Data**: View and filter studies and datasets from our inventory, with interactive knowledge graphs and AI-powered data analysis
    2. **Publications**: Browse and filter publications from PubMed Central with author/affiliation networks and AI-powered publication analysis
    3. **Code**: Explore GitHub repositories related to Alzheimer's studies with repository connection graphs and AI-powered code analysis
    4. **Biorepositories**: Access iNDI iPSC lines with neurodegenerative disease-associated variants and AI-powered genetic analysis

    ## Key Features

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

    ## AI Analysis Capabilities

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

    ## Technical Implementation

    The application is built using:
    - **Streamlit**: For the web interface and data visualization
    - **Pandas**: For data manipulation and filtering
    - **NetworkX**: For knowledge graph creation and analysis
    - **Plotly**: For interactive network visualizations
    - **scikit-learn**: For semantic search using TF-IDF and cosine similarity
    - **Anthropic Claude**: For AI-powered analysis and insights (optional)

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

    ### AI Analysis

    Each tab includes comprehensive AI-powered analysis using Anthropic's Claude model.
    These analyses provide expert-level insights about research patterns, gaps, and strategic recommendations.

    ## Data Sources

    - **Data Inventory**: Comprehensive list of Alzheimer's studies and datasets with metadata
    - **PubMed Central**: Contains information about studies, including titles, authors, affiliations, abstracts, and keywords
    - **GitHub**: Contains information about codebases, including repositories, owners, contributors, languages, and tools used (biomedical relevance filtered)
    - **iNDI Biorepository**: iPSC lines with neurodegenerative disease-associated genetic variants

    ## Command Line Tools

    The project includes command-line tools for data collection:

    1. **PubMed Search Tool** (`pubmed_search_cli.py`):
       ```bash
       python pubmed_search_cli.py --study <study_name> --abbreviation <abbreviation> --output <output_file>
       ```
       Searches PubMed Central for articles related to a specific study.

    2. **GitHub Search Tool** (`github_search_cli.py`):
       ```bash
       python github_search_cli.py --study <study_name> --abbreviation <abbreviation> --output <output_file>
       ```
       Searches GitHub for repositories related to a specific study.

    3. **HuggingFace Search Tool** (`huggingface_search_cli.py`):
       ```bash
       python huggingface_search_cli.py --study <study_name> --abbreviation <abbreviation> --output <output_file>
       ```
       Searches HuggingFace for models related to a specific study.

    ## Setup

    To run this application:

    1. Install dependencies:
       ```bash
       pip install -r requirements.txt
       ```

    2. Set up your Anthropic API key (optional, for AI analysis):
       ```bash
       export ANTHROPIC_API_KEY="your-api-key-here"
       ```
       Or add to `.streamlit/secrets.toml`:
       ```toml
       ANTHROPIC_API_KEY = "your-api-key-here"
       ```

    3. Run the application:
       ```bash
       streamlit run app.py
       ```

    ## Version History

    ### Version 2.0.0 (Current)
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

    ## License

    This project is licensed under the MIT License - see the LICENSE file for details.
    """) 