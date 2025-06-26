# Deployment Guide

This guide will help you deploy the CARD Catalogue to GitHub and Streamlit Cloud.

## Step 1: Prepare Your Repository

### 1.1 Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: CARD Catalogue app"
```

### 1.2 Create GitHub Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click "New repository"
3. Name it `card-catalogue` (or your preferred name)
4. Make it **Public** (required for Streamlit Cloud free tier)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

### 1.3 Push to GitHub
```bash
git remote add origin https://github.com/yourusername/card-catalogue.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Set Up Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### 2.2 Configure App
- **Repository**: Select your `card-catalogue` repository
- **Branch**: `main`
- **Main file path**: `app.py`
- **App URL**: Choose a custom URL (optional)

### 2.3 Deploy
Click "Deploy" and wait for the build to complete.

## Step 3: Configure Secrets (Optional)

If you want to use AI explanations (currently not implemented in the app), you can add your Anthropic API key:

### 3.1 Get API Key
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or sign in
3. Create a new API key

### 3.2 Add to Streamlit Cloud
1. In your Streamlit Cloud app dashboard
2. Go to "Settings" → "Secrets"
3. Add the following:
```toml
ANTHROPIC_API_KEY = "your-actual-api-key-here"
```

## Step 4: Verify Deployment

1. **Check the app**: Visit your Streamlit Cloud URL
2. **Test functionality**: Try searching and filtering data
3. **Test knowledge graphs**: Generate graphs in each tab
4. **Test exports**: Export filtered data and summaries

## Troubleshooting

### Common Issues

#### 1. Build Fails
- **Error**: "Module not found"
- **Solution**: Ensure all dependencies are in `requirements.txt`

#### 2. Data Files Missing
- **Error**: "File not found"
- **Solution**: Make sure all data files are in the repository root

#### 3. Secrets Not Working
- **Error**: "API key not found"
- **Solution**: Check the secrets format in Streamlit Cloud

### Data Files

Make sure these files are in your repository:
- `dataset-inventory-June_20_2025.tab`
- `pubmed_central_20250620_174508.tab`
- `gits_to_reannotate_completed_20250626_120254.tsv`
- `iNDI_inventory_20250620_122423.tab`

### Security Notes

- ✅ **Safe to deploy**: The app doesn't use API keys for core functionality
- ✅ **Public repository**: No sensitive data in the code
- ✅ **Secrets protected**: API keys are stored securely in Streamlit Cloud
- ⚠️ **Data files**: Consider if your data files should be public

## Customization

### Update App Title
Edit `app.py` line 15:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Update Data Sources
Replace the data files with your own:
1. Update file paths in `load_data()` function
2. Adjust column names if needed
3. Update filters to match your data structure

### Custom Domain
For a custom domain:
1. Go to Streamlit Cloud settings
2. Add your domain in "Custom domain"
3. Configure DNS records as instructed

## Maintenance

### Updating the App
1. Make changes locally
2. Test with `streamlit run app.py`
3. Commit and push to GitHub
4. Streamlit Cloud will automatically redeploy

### Monitoring
- Check Streamlit Cloud dashboard for errors
- Monitor app performance
- Review user feedback

## Support

If you encounter issues:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the [Streamlit Cloud troubleshooting guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
3. Open an issue in this repository 