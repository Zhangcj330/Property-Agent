# Property-Agent

A smart property buyer agent that leverages LLMs, web tools, and image processing to recommend properties based on user preferences.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Local Development Setup](#local-development-setup)
- [Technical Architecture](#technical-architecture)
- [Detailed Components](#detailed-components)
- [Security and Privacy](#security-and-privacy)
- [Future Enhancements](#future-enhancements)
- [Technical Documentation](#technical-documentation)

## Overview
This system is a chatbot-powered property recommendation engine that uses Large Language Models (LLMs), web browser tools, APIs, and image processing to suggest properties based on user preferences through natural conversation.

## Features
- Conversational interface for property search
- Smart preference extraction from natural language
- Multi-source property data integration
- Image-based property analysis
- Personalized property recommendations
- Real-time data enrichment

## Local Development Setup

### Prerequisites
- Python (v3.8 or higher)
- pip (latest version)
- Docker (optional, for containerized deployment)

### Installation Steps
1. Clone the repository
```bash
git clone https://github.com/Zhangcj330/valuation-model-UI.git
cd Property-Agent
```

2. Install dependencies
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
pip install -r frontend/requirements.txt
```

3. Environment Setup
```bash
# Copy example environment files
cp .env.example .env
```
Edit the `.env` file with your configuration:
- API keys for property listing services
- Database credentials
- LLM API credentials

4. Start the Development Server
```bash
# Start FastAPI backend
cd backend
uvicorn app.main:app --reload --port 8000

# Start Streamlit frontend (in a new terminal)
cd frontend
streamlit run app.py
```

The application should now be running at `http://localhost:8501`

## Technical Documentation

### System Requirements

#### Functional Requirements
- Chatbot Interface: Users can input queries about property requirements via natural language
- Preferences Extraction: System infers budget, location, size, style, amenities, etc.
- Search & Filter: System searches property listings based on preferences
- Property Data Enrichment: System fetches additional details using web tools or APIs
- Recommendation Ranking: Provide top property options with rationale

#### Technical Requirements
- LLM Integration: Use a Large Language Model for parsing user input and summarizing preferences
- APIs and Browser Tools: Communicate with listing websites
- Image Processing: Analyze property photos to extract structured data
- Data Storage: Maintain property listings, user profiles, and conversation logs

#### Non-Functional Requirements
- Performance: Response time < 2 seconds for typical queries
- Reliability: Handle large volumes of property listings and simultaneous user sessions
- Security & Privacy: Protect user data and comply with regulations

### Architecture

```
+----------------------------+
|       User Interface       |
| (Web/Chatbot/Mobile App)   |
+------------+---------------+
             | User Query
+------------v---------------+
|    LLM-Based Inference     |
|    (NLP + Preference      |
|    Extraction)            |
+------------+---------------+
             | Preferences
+------------v---------------+         +----------------------+
| Property Recommendation    | <-----> | Property Data Cache  |
| & Scoring Engine          |         | (Listings, Metadata) |
+------------+---------------+         +---------+------------+
             | Recommendations                  ^
+------------v---------------+                  |
| External Integrations      | ----------------+
| (APIs, Browser Tools,      |
| Image Processing)          |
+----------------------------+
```

### Technical Stack

| Layer | Technologies |
|-------|--------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | PostgreSQL/MongoDB |
| LLM Integration | OpenAI GPT/Gemini |
| Image Processing | OpenCV, TensorFlow, PyTorch |
| Infrastructure | AWS/GCP/Azure, Docker, Kubernetes |

### Security and Privacy
- Encrypted data storage and transmission
- API authentication and authorization
- GDPR compliance
- User data protection and anonymization
- Secure conversation logging

### Future Enhancements
1. Advanced Personalization
   - ML models for style preference learning
   - Behavioral analysis

2. Real-Time Features
   - Bidding integration
   - Mortgage calculations
   - Banking API integration

3. Extended Functionality
   - AR/VR property tours
   - Predictive pricing
   - Market trend analysis

For detailed technical specifications and implementation details, please refer to the `/docs` directory.