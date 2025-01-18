# Audio_Enhanced_Conversational_AI_Chatbot

## Project Overview
Audio-Enhanced Conversational AI Chatbot is a user friendly chatbot that transforms video or audio content into an interactive, conversational experience by transcribing audio, processing content for context and providing accurate, real-time answers to user questions.

## Reason behind developing the chatbot
People often consume a wealth of information via videos, webinars, and podcasts but lack efficient ways to interact with or query specific information within these media. The challenge lies in converting long-form audio or video content into an interactive experience, allowing users to ask questions and retrieve precise information from the content without manually searching or listening through the entire file.

## Steps to build the project
1. Transcribing audio from Youtube videos.
2. Processing and organizing this content for effective retrieval.
3. Create summary for entire transcript.
4. Enabling user to query and recieve accurate, context-based answers from the transcription using a Large Language Model(LLM).

## Techstack used

1. Python Programming Language
2. LLMs - Google Gemini Pro Model
3. Streamlit
4. Assembly AI
5. Vector Stores - FAISS

## Project Setup

### 1. Clone the Repository.
```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Create a Virtual Environment.
```bash
conda create -n audiochatbot python=3.11 -y
```

### 3. Activate the Virtual Environment.
```bash
conda activate audiochatbot
```

### 4. Install Project Requirements.
```bash
pip install -r requirements.txt
```

### 5. Environment Variables.

#### Create a .env file and add the required API secret credentials.
```ini
ASSEMBLY_AI_KEY = your_api_key
GOOGLE_API_KEY = your_secret_key
```

### 6. How to run the project.
```bash
streamlit run app.py
```