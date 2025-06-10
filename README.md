# LINE Chatbot with Retrieval Augmented Generation

A chatbot project that integrates with LINE and uses Retrieval Augmented Generation (RAG) to provide intelligent, context-aware responses. This project leverages a vector database for document retrieval and supports environment-based configuration.

## Features
- LINE Messaging API integration
- Retrieval Augmented Generation (RAG) for enhanced responses
- Vector database support (e.g., ChromaDB)
- Environment variable configuration via `.env`
- Modular and extensible codebase

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/LINE-Chatbot-with-Retrieval-Augmented-Generation.git
cd LINE-Chatbot-with-Retrieval-Augmented-Generation
```

### 2. Create and Activate a Virtual Environment (Recommended)
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with your LINE API credentials and other required settings. Example:
```env
LINE_CHANNEL_SECRET=your_channel_secret
LINE_CHANNEL_ACCESS_TOKEN=your_access_token
# Add other environment variables as needed
```

### 5. Run the Chatbot
```sh
python main.py
```

## Notes
- The `chroma_db/` directory is ignored by Git (see `.gitignore`).
- Do **not** commit sensitive information or large files to the repository.
- If you need to track large files, consider using [Git LFS](https://git-lfs.github.com/).

## Troubleshooting
- If you see warnings about large files, remove them from your Git history and add them to `.gitignore`.
- For issues with environment variables, ensure your `.env` file is correctly set up.

## License
MIT License 