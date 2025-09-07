# 🌟 Iron Lady AI Chatbot

An intelligent conversational AI assistant for Iron Lady's leadership development programs, featuring advanced intent classification and retrieval-augmented generation (RAG) capabilities.

## 🚀 Features

- **🤖 Smart Intent Classification**: Uses Hugging Face's pre-trained BART model for zero-shot intent detection
- **📚 RAG-Powered Responses**: Retrieves relevant information from Iron Lady's knowledge base
- **🎯 Context-Aware Conversations**: Tailored responses based on user intent
- **💬 Modern UI**: Clean, responsive Gradio interface
- **⚡ Real-Time Processing**: Instant intent detection and response generation

## 🛠️ Technology Stack

- **Frontend**: Gradio (Web UI)
- **Intent Classification**: Hugging Face Transformers (BART-large-mnli)
- **Language Model**: Groq API (Llama-3.1-8b-instant)
- **RAG Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Text Processing**: RecursiveCharacterTextSplitter

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (for model downloads and API calls)
- Groq API key

## 🔧 Installation

### Step 1: Clone or Download the Project

```bash
# Create project directory
mkdir iron_lady_chatbot
cd iron_lady_chatbot
```

### Step 2: Install Dependencies

```bash
pip install gradio transformers torch groq langchain langchain-community langchain-groq faiss-cpu sentence-transformers
```

### Step 3: Set Up Files

Create the following files in your project directory:

1. **`main.py`** - Main application file
2. **`iron_lady_data.txt`** - Knowledge base content

## 📁 Project Structure

```
iron_lady_chatbot/
├── main.py                 # Main application file
├── iron_lady_data.txt      # Knowledge base content
├── README.md              # This file
└── requirements.txt       # Dependencies (optional)
```

## 🚦 Quick Start

### 1. Add Your Data

Create `iron_lady_data.txt` with Iron Lady's information

### 2. Update API Key

In `main.py`, replace the API key with your own:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

### 3. Run the Application

```bash
python main.py
```

The application will start and display:
```
🚀 Loading models...
✅ All systems ready!
Running on local URL:  http://127.0.0.1:7860
```

## 💡 How It Works

### Intent Classification Pipeline

1. **User Input**: User types a message
2. **Intent Detection**: BART model classifies the intent using zero-shot classification
3. **Intent Mapping**: Maps detected intent to specific response strategy
4. **Response Generation**: Uses intent-specific prompts with RAG retrieval

### Supported Intents

| Intent | Example Queries | Response Focus |
|--------|----------------|---------------|
| `greet` | "Hello", "Hi there" | Welcoming, asks how to help |
| `goodbye` | "Bye", "Thank you" | Encouraging farewell |
| `ask_programs` | "What programs do you offer?" | Program details and benefits |
| `ask_mentors` | "Who are the mentors?" | Mentor credentials and expertise |
| `ask_cost` | "How much does it cost?" | Investment value and ROI |
| `ask_results` | "What are the success stories?" | Impact metrics and outcomes |
| `ask_duration` | "How long are programs?" | Timeline and learning format |
| `general` | Any other question | General helpful information |

### RAG Process

1. **Document Loading**: Loads content from `iron_lady_data.txt`
2. **Text Chunking**: Splits content into 1000-character chunks with 100-character overlap
3. **Embedding**: Creates vector embeddings using Sentence Transformers
4. **Vector Storage**: Stores embeddings in FAISS vector database
5. **Retrieval**: Finds top 3 most relevant chunks for each query
6. **Generation**: Combines retrieved context with user query for response

## 🎮 Usage Examples

### Example 1: Program Inquiry
```
User: "What leadership programs do you offer?"
Intent: ask_programs
Response: Focuses on the three flagship programs with unique benefits highlighted
```

### Example 2: Mentor Information
```
User: "Tell me about your coaches"
Intent: ask_mentors  
Response: Emphasizes mentor credentials and industry experience
```

### Example 3: Cost Information
```
User: "How much investment is required?"
Intent: ask_cost
Response: Discusses ROI, value proposition, and suggests contacting for details
```

## 🔍 Debug Mode

The application prints detected intents in the terminal for debugging:

```
🎯 Detected intent: ask_programs
🎯 Detected intent: ask_cost
🎯 Detected intent: greet
```

## ⚙️ Configuration Options

### Intent Classification Confidence

Adjust the confidence threshold in `classify_intent()`:

```python
if confidence > 0.3:  # Lower = more sensitive, Higher = more conservative
    return intent
```

### RAG Parameters

Modify chunk size and overlap in `build_rag()`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Increase for larger chunks
    chunk_overlap=100   # Adjust overlap between chunks
)
```

### Retrieval Settings

Change number of retrieved documents:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # Number of chunks to retrieve
)
```

## 🎨 UI Customization

### Themes

Change the Gradio theme:

```python
with gr.Blocks(theme="soft") as demo:  # Options: default, soft, monochrome
```

### Example Queries

Modify the example queries in the interface:

```python
gr.Examples(
    examples=[
        "Your custom example 1",
        "Your custom example 2",
        # Add more examples
    ],
    inputs=user_msg
)
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Download Errors**
```
Solution: Ensure stable internet connection. Models download automatically on first run.
```

**2. API Key Issues**
```
Error: Invalid API key
Solution: Verify your Groq API key is correct and active.
```

**3. Memory Issues**
```
Error: CUDA out of memory / RAM issues
Solution: The code automatically uses CPU. For large datasets, consider increasing system RAM.
```

**4. Import Errors**
```
Error: ModuleNotFoundError
Solution: Reinstall dependencies: pip install -r requirements.txt
```

### Performance Optimization

**For Faster Loading:**
- Use smaller embedding models: `all-MiniLM-L6-v2` (default) is already optimized
- Reduce chunk size for smaller datasets
- Cache models locally after first download

**For Better Accuracy:**
- Add more diverse training examples in intent labels
- Increase chunk overlap for better context
- Fine-tune confidence thresholds based on your use case



**Built with ❤️ for Iron Lady's mission to empower women leaders**
