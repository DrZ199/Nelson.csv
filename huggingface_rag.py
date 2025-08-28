#!/usr/bin/env python3
"""
Enhanced RAG System with Hugging Face Embeddings
Uses sentence-transformers for semantic search capabilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence-transformers, fall back to manual implementation if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using fallback implementation")

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch not available")

class ManualEmbeddingModel:
    """Fallback embedding model using TF-IDF when HuggingFace models not available"""
    
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """Fit the TF-IDF model on texts"""
        print("Fitting manual embedding model (TF-IDF)...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.vectorizer.transform(texts)
        # Normalize for cosine similarity
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, norm='l2')
        
        return embeddings.toarray()

class HuggingFaceRAGSystem:
    def __init__(self, csv_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize RAG system with Hugging Face embeddings"""
        print("Initializing Hugging Face RAG System...")
        
        # Load the knowledge base
        print(f"Loading knowledge base from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} chunks")
        
        self.model_name = model_name
        self.embeddings = None
        self.embedding_file = f"/project/workspace/embeddings_{model_name.replace('/', '_')}.pkl"
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Loading Hugging Face model: {model_name}")
            try:
                self.model = SentenceTransformer(model_name)
                self.use_manual = False
                print("✅ Hugging Face model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load HF model: {e}")
                self.model = ManualEmbeddingModel()
                self.use_manual = True
        else:
            print("Using manual TF-IDF embedding model")
            self.model = ManualEmbeddingModel()
            self.use_manual = True
    
    def generate_embeddings(self, force_regenerate: bool = False, sample_size: Optional[int] = None):
        """Generate embeddings for all chunks"""
        
        # Check if embeddings already exist
        if os.path.exists(self.embedding_file) and not force_regenerate:
            print(f"Loading existing embeddings from {self.embedding_file}")
            with open(self.embedding_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            return
        
        print("Generating new embeddings...")
        
        # Sample data if requested (for testing with large datasets)
        if sample_size and len(self.df) > sample_size:
            print(f"Using sample of {sample_size} chunks for testing")
            sample_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.df
        
        # Prepare texts
        texts = []
        for idx, row in sample_df.iterrows():
            # Combine title, summary, and text for better embeddings
            combined_text = f"{row['chapter_title']} {row['section_title']} {row['chunk_summary']} {row['chunk_text'][:500]}"
            texts.append(combined_text)
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        if self.use_manual:
            # Fit and encode with manual model
            self.model.fit(texts)
            self.embeddings = self.model.encode(texts)
        else:
            # Use Hugging Face model
            batch_size = 32
            self.embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        # Save embeddings
        print(f"Saving embeddings to {self.embedding_file}")
        with open(self.embedding_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Update dataframe if we used a sample
        if sample_size and len(self.df) > sample_size:
            self.df = sample_df.reset_index(drop=True)
    
    def semantic_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Perform semantic search using embeddings"""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call generate_embeddings() first.")
        
        # Encode query
        if self.use_manual:
            query_embedding = self.model.encode([query])
        else:
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Calculate similarities
        if len(query_embedding.shape) > 1 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                row = self.df.iloc[idx]
                
                result = {
                    'chunk_id': str(row['id']),
                    'similarity': similarity,
                    'page_number': int(row['page_number']),
                    'chapter_title': str(row['chapter_title']),
                    'section_path': str(row['section_heading_path']),
                    'text_preview': str(row['chunk_text'])[:300] + "...",
                    'full_text': str(row['chunk_text']),
                    'summary': str(row['chunk_summary']),
                    'keywords': str(row['keywords']),
                    'confidence_score': float(row['confidence_score']),
                    'citation': self._format_citation(row)
                }
                results.append(result)
        
        return results
    
    def _format_citation(self, row) -> str:
        """Format citation for a chunk"""
        return (f"{row['authors']}. {row['book_title']}, {row['edition']} Edition. "
               f"{row['publisher']}, {row['year']}. Page {row['page_number']}. "
               f"Chapter: {row['chapter_title']}. Section: {row['section_heading_path']}.")
    
    def compare_search_methods(self, query: str, top_k: int = 3) -> Dict:
        """Compare semantic search with traditional TF-IDF"""
        print(f"\n🔍 Comparing search methods for: '{query}'")
        print("=" * 60)
        
        # Semantic search results
        semantic_results = self.semantic_search(query, top_k=top_k)
        
        # Traditional TF-IDF search for comparison
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        texts = self.df['chunk_text'].fillna('').tolist()
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        
        tfidf_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_tfidf_indices = np.argsort(tfidf_similarities)[::-1][:top_k]
        
        tfidf_results = []
        for idx in top_tfidf_indices:
            row = self.df.iloc[idx]
            tfidf_results.append({
                'similarity': float(tfidf_similarities[idx]),
                'page_number': int(row['page_number']),
                'chapter': str(row['chapter_title']),
                'summary': str(row['chunk_summary'])
            })
        
        return {
            'query': query,
            'semantic_results': semantic_results,
            'tfidf_results': tfidf_results
        }
    
    def medical_query_demo(self):
        """Demonstrate medical queries"""
        medical_queries = [
            "type 1 diabetes treatment in children",
            "fever management in newborn infants",
            "childhood asthma diagnosis and therapy",
            "congenital heart defects screening",
            "pediatric vaccination recommendations",
            "growth failure in toddlers",
            "seizures in adolescents"
        ]
        
        print("\n🏥 MEDICAL QUERY DEMONSTRATION")
        print("=" * 60)
        
        for query in medical_queries[:3]:  # Demo first 3 queries
            results = self.semantic_search(query, top_k=2)
            
            print(f"\n📋 Query: {query}")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"\n🔸 Result {i} (Similarity: {result['similarity']:.3f})")
                print(f"   📍 Page {result['page_number']} | {result['chapter_title']}")
                print(f"   📝 {result['summary']}")
                print(f"   🔗 {result['citation'][:100]}...")
    
    def save_search_index(self, filename: str = "/project/workspace/hf_search_index.json"):
        """Save search index for later use"""
        if self.embeddings is None:
            print("No embeddings to save")
            return
        
        # Save metadata and index info
        index_data = {
            'model_name': self.model_name,
            'num_chunks': len(self.df),
            'embedding_shape': self.embeddings.shape,
            'created_at': datetime.now().isoformat(),
            'use_manual': self.use_manual
        }
        
        with open(filename, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Search index metadata saved to {filename}")

def demo_hugging_face_rag():
    """Run demonstration of Hugging Face RAG system"""
    csv_path = "/project/workspace/nelson_chunks.csv"
    
    # Initialize system
    print("🤗 HUGGING FACE RAG SYSTEM DEMO")
    print("=" * 60)
    
    # Choose model based on availability
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        model_name = 'all-MiniLM-L6-v2'  # Fast, good general model
        # Alternative medical models you could try:
        # 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        # 'dmis-lab/biobert-base-cased-v1.1'
    else:
        model_name = 'manual-tfidf'
    
    rag_system = HuggingFaceRAGSystem(csv_path, model_name)
    
    # Generate embeddings (use sample for demo due to size)
    sample_size = 1000  # Use 1000 chunks for demo
    rag_system.generate_embeddings(sample_size=sample_size)
    
    # Run medical query demo
    rag_system.medical_query_demo()
    
    # Compare search methods
    comparison = rag_system.compare_search_methods(
        "diabetes treatment in pediatric patients"
    )
    
    print(f"\n📊 SEARCH METHOD COMPARISON")
    print("=" * 60)
    print(f"Query: {comparison['query']}")
    
    print(f"\n🤗 Semantic Search (Hugging Face):")
    for i, result in enumerate(comparison['semantic_results'][:2], 1):
        print(f"  {i}. Page {result['page_number']} (sim: {result['similarity']:.3f})")
        print(f"     {result['summary'][:100]}...")
    
    print(f"\n📊 Traditional TF-IDF Search:")
    for i, result in enumerate(comparison['tfidf_results'][:2], 1):
        print(f"  {i}. Page {result['page_number']} (sim: {result['similarity']:.3f})")
        print(f"     {result['summary'][:100]}...")
    
    # Save search index
    rag_system.save_search_index()
    
    return rag_system

def create_production_embeddings():
    """Create embeddings for the full dataset"""
    print("\n🏭 CREATING PRODUCTION EMBEDDINGS")
    print("=" * 60)
    
    csv_path = "/project/workspace/nelson_chunks.csv"
    model_name = 'all-MiniLM-L6-v2' if SENTENCE_TRANSFORMERS_AVAILABLE else 'manual-tfidf'
    
    rag_system = HuggingFaceRAGSystem(csv_path, model_name)
    
    # Generate embeddings for all chunks (no sampling)
    rag_system.generate_embeddings(force_regenerate=False)
    
    print(f"✅ Production embeddings ready!")
    print(f"   Model: {model_name}")
    print(f"   Chunks: {len(rag_system.df)}")
    print(f"   Embedding shape: {rag_system.embeddings.shape}")
    
    return rag_system

if __name__ == "__main__":
    # Run demo with sample
    demo_system = demo_hugging_face_rag()
    
    print(f"\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    print("1. For production use, run: create_production_embeddings()")
    print("2. Install sentence-transformers for better performance:")
    print("   pip install sentence-transformers")
    print("3. Consider using medical-specific models like:")
    print("   - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    print("   - dmis-lab/biobert-base-cased-v1.1")