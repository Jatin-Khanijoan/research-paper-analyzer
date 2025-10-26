import PyPDF2
import nltk
import string
import re
import os
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from typing import List, Dict, Tuple
import pandas as pd

# ================================================================
# NLTK SETUP
# ================================================================
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    required = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for res in required:
        try:
            if res == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif res in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{res}')
            else:
                nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f"Downloading {res}...")
            nltk.download(res, quiet=True)

ensure_nltk_resources()

# ================================================================
# MODEL INITIALIZATION
# ================================================================
@st.cache_resource
def load_models():
    """Load embedding model and initialize Gemini"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Gemini
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
        st.stop()
    
    genai.configure(api_key=api_key)
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-thinking-exp-1219')
    gemini_model = genai.GenerativeModel(model_name)
    
    return embedding_model, gemini_model

# ================================================================
# NLP PREPROCESSING
# ================================================================
class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text: str) -> str:
        """Advanced preprocessing with lemmatization"""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]
        return " ".join(tokens)
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using POS tagging and capitalization patterns"""
        sentences = sent_tokenize(text)
        entities = {'PERSON': [], 'ORG': [], 'CONCEPT': []}
        
        # Pattern for author names (capitalized words before year or in citations)
        author_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:et al\.|and [A-Z]|\(\d{4}\))'
        
        for sentence in sentences[:50]:  # First 50 sentences for efficiency
            # Find potential authors
            authors = re.findall(author_pattern, sentence)
            entities['PERSON'].extend(authors)
            
            # Find capitalized multi-word terms (potential concepts/organizations)
            cap_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b'
            concepts = re.findall(cap_pattern, sentence)
            
            for concept in concepts:
                if any(keyword in concept.lower() for keyword in ['university', 'institute', 'lab', 'department']):
                    entities['ORG'].append(concept)
                elif len(concept.split()) > 1:  # Multi-word capitalized terms
                    entities['CONCEPT'].append(concept)
        
        # Deduplicate and limit
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]
        
        return entities
    
    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = scores.argsort()[-top_n:][::-1]
            
            keywords = [(feature_names[i], scores[i]) for i in top_indices]
            return keywords
        except:
            return []
    
    def extract_methodology_section(self, text: str) -> str:
        """Extract methodology section using section headers"""
        patterns = [
            r'(?:^|\n)((?:Methodology|Methods|Experimental Setup|Approach|Materials and Methods).*?)(?=\n(?:[A-Z][a-z]+ ?){1,3}\n|\n\d+\.|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                method_text = match.group(1)
                # Limit to reasonable length
                sentences = sent_tokenize(method_text)
                return ' '.join(sentences[:10])
        
        return "Methodology section not clearly identified."
    
    def extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings using linguistic patterns"""
        # Look for results/conclusion sections
        result_pattern = r'(?:^|\n)((?:Results?|Findings?|Conclusions?).*?)(?=\n(?:[A-Z][a-z]+ ?){1,3}\n|\n\d+\.|\Z)'
        
        findings = []
        match = re.search(result_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            result_text = match.group(1)
            sentences = sent_tokenize(result_text)
            
            # Find sentences with findings indicators
            finding_keywords = ['found', 'showed', 'demonstrated', 'revealed', 'indicated', 
                               'suggested', 'observed', 'discovered', 'proved', 'concluded']
            
            for sent in sentences[:20]:
                if any(keyword in sent.lower() for keyword in finding_keywords):
                    if len(sent.split()) > 10:  # Substantial sentences
                        findings.append(sent.strip())
                        if len(findings) >= 5:
                            break
        
        return findings if findings else ["Key findings section not clearly identified."]

# ================================================================
# PDF PROCESSING
# ================================================================
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file) -> str:
        """Extract text from PDF"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            if page:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        """Extract paper metadata using regex patterns"""
        metadata = {
            'title': 'Unknown',
            'abstract': 'Not found',
            'year': 'Unknown'
        }
        
        # Extract title (first non-empty line, usually all caps or title case)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            metadata['title'] = lines[0][:200]  # Limit length
        
        # Extract abstract
        abstract_pattern = r'(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.*?)(?=\n\s*\n|\nIntroduction|\n1\.|\nKeywords:)'
        abstract_match = re.search(abstract_pattern, text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            metadata['abstract'] = ' '.join(abstract.split()[:200])  # Limit words
        
        # Extract year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text[:2000])  # Search in first 2000 chars
        if years:
            metadata['year'] = years[0]
        
        return metadata
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
        """Intelligent chunking by paragraphs and sentences"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            for sent in sentences:
                words = sent.split()
                if current_size + len(words) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_size = len(words)
                else:
                    current_chunk.append(sent)
                    current_size += len(words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# ================================================================
# VECTOR SEARCH
# ================================================================
class VectorSearch:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.nlp_processor = NLPProcessor()
        
    def build_index(self, chunks: List[str]) -> Tuple[faiss.Index, List[str]]:
        """Build FAISS index from text chunks"""
        processed_chunks = [self.nlp_processor.preprocess(c) for c in chunks]
        embeddings = self.embedding_model.encode(
            processed_chunks, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype('float32')
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return index, chunks
    
    def search(self, query: str, index: faiss.Index, chunks: List[str], top_k: int = 5) -> List[str]:
        """Search for relevant chunks"""
        processed_query = self.nlp_processor.preprocess(query)
        query_embedding = self.embedding_model.encode(
            [processed_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype('float32')
        
        D, I = index.search(query_embedding, top_k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(chunks):
                results.append(chunks[idx])
        
        return results

# ================================================================
# CITATION NETWORK ANALYSIS
# ================================================================
def analyze_citations(texts: List[str]) -> Dict:
    """Analyze citation patterns across papers"""
    citation_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)'
    
    all_citations = []
    for text in texts:
        citations = re.findall(citation_pattern, text[:5000])  # First 5000 chars
        all_citations.extend([f"{author} ({year})" for author, year in citations])
    
    citation_counts = Counter(all_citations)
    
    return {
        'total_citations': len(all_citations),
        'unique_citations': len(citation_counts),
        'top_cited': citation_counts.most_common(10)
    }

# ================================================================
# COMPARATIVE ANALYSIS
# ================================================================
def compare_papers(papers_data: List[Dict], gemini_model) -> str:
    """Generate comparative analysis of multiple papers"""
    if len(papers_data) < 2:
        return "Need at least 2 papers for comparison."
    
    comparison_prompt = f"""You are a research analyst. Compare the following {len(papers_data)} research papers:

"""
    
    for i, paper in enumerate(papers_data, 1):
        comparison_prompt += f"""
Paper {i}:
Title: {paper['metadata']['title']}
Year: {paper['metadata']['year']}
Key Terms: {', '.join([kw[0] for kw in paper['keywords'][:5]])}
Abstract: {paper['metadata']['abstract'][:300]}...

"""
    
    comparison_prompt += """
Provide a structured comparison covering:
1. Common themes and research gaps
2. Methodological differences
3. Chronological evolution (if applicable)
4. Complementary insights
5. Conflicting findings (if any)

Keep it concise and academic.
"""
    
    try:
        response = gemini_model.generate_content(comparison_prompt)
        return response.text
    except Exception as e:
        return f"Error generating comparison: {str(e)}"

# ================================================================
# STREAMLIT UI
# ================================================================
def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide", page_icon="📚")
    
    st.title("📚 Research Paper Analyzer with NLP")
    st.markdown("*Advanced NLP-powered tool for analyzing research papers*")
    
    # Load models
    embedding_model, gemini_model = load_models()
    nlp_processor = NLPProcessor()
    pdf_processor = PDFProcessor()
    vector_search = VectorSearch(embedding_model)
    
    # Sidebar
    st.sidebar.header("📤 Upload Papers")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF research papers", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 NLP Features")
    st.sidebar.markdown("""
    - Named Entity Recognition
    - TF-IDF Keyword Extraction
    - Semantic Search (FAISS)
    - Citation Network Analysis
    - POS Tagging
    - Text Summarization
    - Comparative Analysis
    """)
    
    if not uploaded_files:
        st.info("👈 Upload research papers (PDF) to begin analysis")
        st.markdown("### Features:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - 📄 Multi-paper upload
            - 🔍 Semantic search across papers
            - 🏷️ Automatic keyword extraction
            - 👥 Named entity recognition
            """)
        with col2:
            st.markdown("""
            - 📊 Citation analysis
            - 🔬 Methodology extraction
            - 🎯 Key findings summary
            - 📈 Comparative analysis
            """)
        return
    
    # Process uploaded papers
    if 'papers_data' not in st.session_state or st.sidebar.button("🔄 Reprocess Papers"):
        with st.spinner("Processing papers with NLP..."):
            papers_data = []
            
            for pdf_file in uploaded_files:
                # Extract text
                text = pdf_processor.extract_text(pdf_file)
                
                # Extract metadata
                metadata = pdf_processor.extract_metadata(text)
                
                # Chunk text
                chunks = pdf_processor.chunk_text(text)
                
                # Build search index
                index, chunks = vector_search.build_index(chunks)
                
                # Extract keywords using TF-IDF
                keywords = nlp_processor.extract_keywords_tfidf([text])
                
                # Extract named entities
                entities = nlp_processor.extract_named_entities(text)
                
                # Extract methodology
                methodology = nlp_processor.extract_methodology_section(text)
                
                # Extract findings
                findings = nlp_processor.extract_key_findings(text)
                
                papers_data.append({
                    'filename': pdf_file.name,
                    'text': text,
                    'chunks': chunks,
                    'index': index,
                    'metadata': metadata,
                    'keywords': keywords,
                    'entities': entities,
                    'methodology': methodology,
                    'findings': findings
                })
            
            st.session_state.papers_data = papers_data
            st.success(f"✅ Processed {len(papers_data)} papers successfully!")
    
    papers_data = st.session_state.papers_data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Search & Q&A", 
        "🏷️ NLP Analysis", 
        "📈 Comparative Analysis",
        "📚 Citation Network"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Papers Overview")
        
        for i, paper in enumerate(papers_data, 1):
            with st.expander(f"📄 Paper {i}: {paper['filename']}", expanded=(i==1)):
                st.markdown(f"**Title:** {paper['metadata']['title']}")
                st.markdown(f"**Year:** {paper['metadata']['year']}")
                st.markdown(f"**Chunks:** {len(paper['chunks'])}")
                
                st.markdown("**Abstract:**")
                st.write(paper['metadata']['abstract'])
                
                st.markdown("**Top Keywords (TF-IDF):**")
                keywords_str = ", ".join([f"{kw[0]} ({kw[1]:.3f})" for kw in paper['keywords'][:10]])
                st.write(keywords_str)
    
    # TAB 2: Search & Q&A
    with tab2:
        st.header("🔍 Semantic Search & Question Answering")
        
        selected_papers = st.multiselect(
            "Select papers to search:",
            options=[p['filename'] for p in papers_data],
            default=[p['filename'] for p in papers_data]
        )
        
        query = st.text_input("Ask a question or search for concepts:")
        
        if query and st.button("Search"):
            with st.spinner("Searching across papers..."):
                all_results = []
                
                for paper in papers_data:
                    if paper['filename'] in selected_papers:
                        results = vector_search.search(query, paper['index'], paper['chunks'], top_k=3)
                        for result in results:
                            all_results.append({
                                'paper': paper['filename'],
                                'text': result
                            })
                
                if all_results:
                    st.success(f"Found {len(all_results)} relevant passages")
                    
                    # Generate answer using Gemini
                    context = "\n\n".join([f"From {r['paper']}:\n{r['text']}" for r in all_results[:5]])
                    
                    prompt = f"""Based on the following research paper excerpts, answer this question: {query}

Context:
{context}

Provide a concise, academic answer citing which papers support your claims.
"""
                    
                    try:
                        with st.spinner("Generating answer..."):
                            response = gemini_model.generate_content(prompt)
                            st.markdown("### 🤖 AI-Generated Answer")
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("### 📄 Source Passages")
                    for i, result in enumerate(all_results[:5], 1):
                        with st.expander(f"Source {i}: {result['paper']}"):
                            st.write(result['text'])
                else:
                    st.warning("No relevant passages found.")
    
    # TAB 3: NLP Analysis
    with tab3:
        st.header("🏷️ Advanced NLP Analysis")
        
        selected_paper_idx = st.selectbox(
            "Select paper for detailed NLP analysis:",
            options=range(len(papers_data)),
            format_func=lambda x: papers_data[x]['filename']
        )
        
        paper = papers_data[selected_paper_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Named Entities")
            
            st.markdown("**👥 People/Authors:**")
            if paper['entities']['PERSON']:
                st.write(", ".join(paper['entities']['PERSON'][:10]))
            else:
                st.write("None detected")
            
            st.markdown("**🏢 Organizations:**")
            if paper['entities']['ORG']:
                st.write(", ".join(paper['entities']['ORG'][:10]))
            else:
                st.write("None detected")
            
            st.markdown("**💡 Key Concepts:**")
            if paper['entities']['CONCEPT']:
                st.write(", ".join(paper['entities']['CONCEPT'][:10]))
            else:
                st.write("None detected")
        
        with col2:
            st.subheader("Keywords (TF-IDF)")
            keywords_df = pd.DataFrame(
                paper['keywords'][:15],
                columns=['Keyword', 'TF-IDF Score']
            )
            st.dataframe(keywords_df, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("🔬 Methodology Section")
        st.write(paper['methodology'])
        
        st.markdown("---")
        
        st.subheader("🎯 Key Findings")
        for i, finding in enumerate(paper['findings'][:5], 1):
            st.markdown(f"{i}. {finding}")
    
    # TAB 4: Comparative Analysis
    with tab4:
        st.header("📈 Comparative Analysis")
        
        if len(papers_data) < 2:
            st.warning("Upload at least 2 papers for comparative analysis.")
        else:
            if st.button("Generate Comparison"):
                with st.spinner("Analyzing papers..."):
                    comparison = compare_papers(papers_data, gemini_model)
                    st.markdown(comparison)
            
            st.markdown("---")
            st.subheader("Keyword Overlap Analysis")
            
            # Create keyword overlap matrix
            all_keywords = {}
            for i, paper in enumerate(papers_data):
                all_keywords[paper['filename']] = set([kw[0] for kw in paper['keywords'][:20]])
            
            overlap_data = []
            papers_names = [p['filename'] for p in papers_data]
            
            for i, paper1 in enumerate(papers_names):
                row = []
                for paper2 in papers_names:
                    overlap = len(all_keywords[paper1] & all_keywords[paper2])
                    row.append(overlap)
                overlap_data.append(row)
            
            overlap_df = pd.DataFrame(
                overlap_data,
                index=papers_names,
                columns=papers_names
            )
            
            st.dataframe(overlap_df, use_container_width=True)
            st.caption("Numbers represent shared keywords between papers")
    
    # TAB 5: Citation Network
    with tab5:
        st.header("📚 Citation Network Analysis")
        
        with st.spinner("Analyzing citations..."):
            all_texts = [p['text'] for p in papers_data]
            citation_data = analyze_citations(all_texts)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Citations", citation_data['total_citations'])
            with col2:
                st.metric("Unique Works Cited", citation_data['unique_citations'])
            with col3:
                avg_citations = citation_data['total_citations'] / len(papers_data) if papers_data else 0
                st.metric("Avg Citations per Paper", f"{avg_citations:.1f}")
            
            st.markdown("---")
            st.subheader("Top Cited Works")
            
            if citation_data['top_cited']:
                citation_df = pd.DataFrame(
                    citation_data['top_cited'],
                    columns=['Citation', 'Count']
                )
                st.dataframe(citation_df, use_container_width=True)
            else:
                st.info("No citations detected in standard format (Author, Year)")

if __name__ == "__main__":
    main()