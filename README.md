# 📚 Research Paper Analyzer with Advanced NLP

An intelligent research paper analysis tool powered by Natural Language Processing (NLP) and AI. Upload multiple research papers (PDF) and get instant insights through semantic search, keyword extraction, named entity recognition, citation analysis, and AI-powered comparative analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Features

### 🧠 Advanced NLP Techniques
- **Named Entity Recognition (NER)** - Automatically extracts authors, organizations, and key concepts
- **TF-IDF Keyword Extraction** - Identifies statistically significant terms across papers
- **Semantic Search with FAISS** - Vector similarity search using sentence embeddings
- **Text Preprocessing Pipeline** - Tokenization, lemmatization, stopword removal
- **Citation Network Analysis** - Extracts and analyzes citation patterns
- **Methodology Extraction** - Automatically identifies research methods sections
- **Key Findings Extraction** - Detects and summarizes important findings

### 🔍 Core Functionality
- ✅ Upload multiple research papers (PDF format)
- ✅ Semantic question answering across all papers
- ✅ AI-powered comparative analysis
- ✅ Citation network visualization
- ✅ Keyword overlap analysis between papers
- ✅ Automatic metadata extraction (title, abstract, year)
- ✅ Intelligent text chunking for efficient search

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (free tier available)

### 1. Clone the Repository
```bash
git clone https://github.com/JatinKhanijoan/research-paper-analyzer.git
cd research-paper-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```properties
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-thinking-exp-1219
```

**Option B: Environment Variables**

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your_gemini_api_key_here
set GEMINI_MODEL=gemini-2.0-flash-thinking-exp-1219
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"
$env:GEMINI_MODEL="gemini-2.0-flash-thinking-exp-1219"
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="gemini-2.0-flash-thinking-exp-1219"
```

### 4. Get Your Free Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key and add it to your `.env` file or environment variables

### 5. Run the Application
```bash
streamlit run main.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload Papers
- Click "Upload PDF research papers" in the sidebar
- Select one or multiple PDF files
- Wait for processing to complete

### 2. Explore the Tabs

#### 📊 Overview Tab
- View all uploaded papers
- See extracted metadata (title, year, abstract)
- Check top keywords identified by TF-IDF

#### 🔍 Search & Q&A Tab
- Select papers to search
- Ask natural language questions
- Get AI-generated answers with source citations
- View relevant passages from papers

#### 🏷️ NLP Analysis Tab
- **Named Entities**: Extracted authors, organizations, concepts
- **Keywords**: TF-IDF ranked important terms
- **Methodology**: Automatically extracted research methods
- **Key Findings**: Important results and conclusions

#### 📈 Comparative Analysis Tab
- Generate AI-powered comparison of multiple papers
- View keyword overlap matrix
- Identify common themes and differences

#### 📚 Citation Network Tab
- See total citations across papers
- View most frequently cited works
- Analyze citation patterns

---

## 🛠️ Technical Architecture

### NLP Pipeline

```
PDF Upload
    ↓
Text Extraction (PyPDF2)
    ↓
Preprocessing (Tokenization, Lemmatization, Stopword Removal)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│   TF-IDF        │  Named Entity    │   Semantic      │
│   Keyword       │  Recognition     │   Embeddings    │
│   Extraction    │  (Regex + POS)   │   (all-MiniLM)  │
└─────────────────┴──────────────────┴─────────────────┘
    ↓                    ↓                    ↓
Keywords            Entities            FAISS Index
    ↓                    ↓                    ↓
        AI-Powered Analysis (Gemini)
                    ↓
            User Interface (Streamlit)
```

### Key Technologies

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web interface and user interaction |
| **PyPDF2** | PDF text extraction |
| **NLTK** | Tokenization, POS tagging, lemmatization |
| **Sentence Transformers** | Semantic embeddings (all-MiniLM-L6-v2) |
| **FAISS** | Efficient vector similarity search |
| **Scikit-learn** | TF-IDF vectorization |
| **Google Gemini** | AI-powered analysis and Q&A |
| **NetworkX** | Citation network analysis |
| **Pandas** | Data manipulation and display |

---

## 📁 Project Structure

```
research-paper-analyzer/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create this)
├── README.md              # This file
│
└── (Auto-generated folders)
    ├── .streamlit/        # Streamlit config (optional)
    └── nltk_data/         # NLTK resources (auto-downloaded)
```

---

## 🎓 NLP Techniques Explained

### 1. Text Preprocessing
- **Tokenization**: Splits text into words and sentences
- **Lemmatization**: Reduces words to base form (running → run)
- **Stopword Removal**: Removes common words (the, is, at)

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
- Identifies important words by balancing frequency and uniqueness
- Higher scores = more important/distinctive terms

### 3. Semantic Embeddings
- Converts text to 384-dimensional vectors
- Captures meaning, not just keywords
- Enables "smart" search based on concept similarity

### 4. Named Entity Recognition
- Uses regex patterns and POS tagging
- Identifies proper nouns (authors, organizations)
- Extracts multi-word concepts

### 5. FAISS Vector Search
- Fast similarity search in high-dimensional space
- Inner product similarity (cosine similarity with normalized vectors)
- Retrieves semantically similar text chunks

---

## 🔧 Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY not found"**
```bash
# Check if variable is set
echo $GOOGLE_API_KEY  # Linux/Mac
echo %GOOGLE_API_KEY%  # Windows CMD
echo $env:GOOGLE_API_KEY  # Windows PowerShell

# If empty, set it in .env file or environment
```

**2. "Missing ScriptRunContext" warnings**
- You're running with `python main.py` instead of `streamlit run main.py`
- Always use: `streamlit run main.py`

**3. Module not found errors**
```bash
pip install -r requirements.txt
```

**4. NLTK resources not downloading**
- The app auto-downloads them on first run
- If it fails, manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

**5. Port already in use**
```bash
streamlit run main.py --server.port 8502
```

**6. PDF extraction fails**
- Ensure PDF is not encrypted
- Try a different PDF
- Check if PDF contains extractable text (not just scanned images)

---

## 🚀 Future Enhancements

- [ ] Add support for DOCX and TXT files
- [ ] Implement advanced citation graph visualization
- [ ] Add export functionality (PDF reports, CSV data)
- [ ] Multi-language support
- [ ] Custom NER model training
- [ ] Batch processing for 50+ papers
- [ ] Integration with academic databases (arXiv, PubMed)
- [ ] Collaborative annotation features
- [ ] Timeline visualization of research evolution

---

## 📊 Performance Notes

- **Processing Time**: ~2-5 seconds per paper (depending on length)
- **Memory Usage**: ~500MB base + ~50MB per paper
- **Optimal Paper Count**: 2-10 papers for best UX
- **Max Paper Size**: Works well up to 100 pages per PDF

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Improve NER accuracy
- Add more visualization options
- Optimize FAISS indexing
- Enhance methodology extraction
- Add unit tests

---

## 🙏 Acknowledgments

- **Sentence Transformers** - For semantic embeddings
- **FAISS** - For efficient similarity search
- **Google Gemini** - For AI-powered analysis
- **Streamlit** - For rapid UI development
- **NLTK** - For comprehensive NLP toolkit

---

## 📚 References

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [NLTK Book](https://www.nltk.org/book/)

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search [existing issues](https://github.com/JatinKhanijoan/research-paper-analyzer/issues)
3. Create a [new issue](https://github.com/JatinKhanijoan/research-paper-analyzer/issues/new) with:
   - Python version
   - Error message
   - Steps to reproduce

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=JatinKhanijoan/research-paper-analyzer&type=Date)](https://star-history.com/JatinKhanijoan/research-paper-analyzer&Date)

---

<div align="center">

Made with ❤️ for the research community

**[⬆ Back to Top](#-research-paper-analyzer-with-advanced-nlp)**

</div>
