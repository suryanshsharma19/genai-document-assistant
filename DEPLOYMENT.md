# Deployment Guide for GenAI Document Assistant

## Streamlit Cloud Deployment with Backend

This guide explains how to deploy the GenAI Document Assistant to Streamlit Cloud while maintaining the backend functionality.

### Prerequisites

1. **GitHub Repository**: Your code is already on GitHub at `suryanshsharma19/genai-document-assistant`
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Backend Hosting**: You'll need to host the backend separately

### Option 1: Full Backend + Frontend Deployment

#### Step 1: Deploy Backend to Railway/Render/Heroku

**Railway (Recommended - Free tier available):**

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub account
3. Create new project from GitHub repository
4. Set environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```
5. Deploy the backend
6. Get your backend URL (e.g., `https://your-app.railway.app`)

#### Step 2: Deploy Frontend to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select repository: `suryanshsharma19/genai-document-assistant`
4. Set the main file path: `frontend/app.py`
5. Add environment variable:
   ```
   BACKEND_URL=https://your-app.railway.app
   ```
6. Deploy

### Option 2: Docker Deployment (Recommended)

#### Using Railway with Docker

1. **Backend Deployment:**
   ```bash
   # Railway will use the Dockerfile.backend
   # Set environment variables in Railway dashboard
   ```

2. **Frontend Deployment:**
   - Use Streamlit Cloud
   - Set `BACKEND_URL` to your Railway backend URL

### Option 3: Local Development

```bash
# Run locally
python start.py
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

### Environment Variables for Production

#### Backend (.env):
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DATABASE_URL=your_database_url
REDIS_URL=your_redis_url
SECRET_KEY=your_secret_key
DEBUG=false
```

#### Frontend (Streamlit Cloud):
```bash
BACKEND_URL=https://your-backend-url.com
```

### Deployment Checklist

- [ ] Backend deployed and accessible
- [ ] Environment variables configured
- [ ] Frontend deployed to Streamlit Cloud
- [ ] CORS configured (if needed)
- [ ] API keys secured
- [ ] Database configured
- [ ] File uploads working
- [ ] AI services responding

### Troubleshooting

1. **CORS Issues**: Ensure backend allows requests from Streamlit Cloud domain
2. **API Timeouts**: Increase timeout settings for large documents
3. **File Upload Issues**: Check file size limits and storage configuration
4. **Environment Variables**: Verify all required variables are set

### Cost Considerations

- **Streamlit Cloud**: Free tier available
- **Railway**: Free tier for small projects
- **OpenAI API**: Pay-per-use
- **Database**: Free tiers available on Railway/Render

### Security Notes

- Never commit API keys to GitHub
- Use environment variables for all secrets
- Enable HTTPS for production
- Implement proper authentication if needed 