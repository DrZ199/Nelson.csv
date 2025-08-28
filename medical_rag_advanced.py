#!/usr/bin/env python3
"""
Medical-Optimized RAG System with Enhanced Embeddings
Specialized for pediatric medical content with intelligent fallbacks
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Optional
import json
import pickle
import os
import re
from datetime import datetime
from collections import Counter

class MedicalEmbeddingSystem:
    """Advanced medical text embedding system with domain-specific optimizations"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.is_fitted = False
        
        # Medical vocabulary enhancement
        self.medical_terms = [
            'diabetes', 'asthma', 'fever', 'treatment', 'therapy', 'diagnosis', 'syndrome',
            'disease', 'disorder', 'patient', 'pediatric', 'infant', 'child', 'adolescent',
            'medication', 'drug', 'symptom', 'clinical', 'pathology', 'physiology',
            'congenital', 'genetic', 'infectious', 'autoimmune', 'malignant', 'benign'
        ]
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for medical terms
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            vocabulary=None
        )
        
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.embeddings_cache = {}
    
    def _enhance_vocabulary(self, texts: List[str]) -> Dict[str, float]:
        """Extract and weight medical vocabulary"""
        print("Extracting medical vocabulary...")
        
        # Combine all texts for vocabulary analysis
        all_text = ' '.join(texts).lower()
        
        # Medical term patterns
        medical_patterns = {
            r'\b\w*itis\b': 2.0,      # Inflammations (arthritis, dermatitis)
            r'\b\w*osis\b': 2.0,      # Conditions (fibrosis, acidosis)  
            r'\b\w*emia\b': 2.0,      # Blood conditions (anemia, leukemia)
            r'\b\w*pathy\b': 2.0,     # Diseases (nephropathy, neuropathy)
            r'\b\w*therapy\b': 1.5,   # Treatments (chemotherapy, physiotherapy)
            r'\b\w*syndrome\b': 2.0,  # Syndromes
            r'\b\w*genic\b': 1.5,     # Origins (carcinogenic, pathogenic)
        }
        
        vocabulary_weights = {}
        
        # Extract pattern-based terms
        for pattern, weight in medical_patterns.items():
            matches = re.findall(pattern, all_text)
            for match in matches:
                if len(match) > 4:  # Filter short matches
                    vocabulary_weights[match] = weight
        
        # Add predefined medical terms
        for term in self.medical_terms:
            if term in all_text:
                vocabulary_weights[term] = 2.0
        
        print(f"Enhanced vocabulary with {len(vocabulary_weights)} medical terms")
        return vocabulary_weights
    
    def fit(self, texts: List[str]):
        """Fit the medical embedding model"""
        print("Fitting medical embedding model...")
        
        # Enhance texts with medical context
        enhanced_texts = []
        for text in texts:
            # Extract medical terms and boost their importance
            enhanced_text = self._enhance_medical_text(text)
            enhanced_texts.append(enhanced_text)
        
        # Fit TF-IDF with enhanced vocabulary
        print("Training TF-IDF vectorizer...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(enhanced_texts)
        
        # Apply dimensionality reduction with SVD (LSA)
        print(f"Applying SVD to reduce to {self.embedding_dim} dimensions...")
        self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
        print("✅ Medical embedding model fitted successfully")
    
    def _enhance_medical_text(self, text: str) -> str:
        """Enhance text by emphasizing medical terms"""
        # Duplicate important medical terms to boost their TF-IDF scores
        enhanced_text = text
        
        for term in self.medical_terms:
            if term in text.lower():
                # Add the term multiple times to boost its importance
                enhanced_text += f" {term} {term}"
        
        # Extract and emphasize medical patterns
        medical_pattern_terms = re.findall(r'\b\w*(?:itis|osis|emia|pathy|therapy|syndrome|genic)\b', 
                                          text.lower())
        for term in medical_pattern_terms:
            if len(term) > 4:
                enhanced_text += f" {term} {term}"
        
        return enhanced_text
    
    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Encode texts to medical embeddings"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Process in batches to manage memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Enhance medical context
            enhanced_batch = [self._enhance_medical_text(text) for text in batch_texts]
            
            # Transform to TF-IDF
            tfidf_batch = self.tfidf_vectorizer.transform(enhanced_batch)
            
            # Apply SVD transformation
            svd_embeddings = self.svd.transform(tfidf_batch)
            
            # Normalize embeddings
            normalized_embeddings = normalize(svd_embeddings, norm='l2')
            
            all_embeddings.append(normalized_embeddings)
        
        # Combine all batches
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings

class AdvancedMedicalRAG:
    """Advanced RAG system optimized for medical/pediatric content"""
    
    def __init__(self, csv_path: str, embedding_dim: int = 512):
        print("🏥 Initializing Advanced Medical RAG System...")
        
        # Load knowledge base
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} medical chunks")
        
        # Initialize medical embedding system
        self.embedding_model = MedicalEmbeddingSystem(embedding_dim)
        self.embeddings = None
        self.embedding_file = f"/project/workspace/medical_embeddings_{embedding_dim}d.pkl"
        
        # Medical specialties for context
        self.medical_specialties = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'arrhythmia', 'murmur'],
            'endocrinology': ['diabetes', 'hormone', 'thyroid', 'growth', 'insulin'],
            'pulmonology': ['asthma', 'respiratory', 'lung', 'breathing', 'pneumonia'],
            'infectious_disease': ['fever', 'infection', 'bacteria', 'virus', 'antibiotic'],
            'neurology': ['seizure', 'neurologic', 'brain', 'development', 'cognitive'],
            'gastroenterology': ['diarrhea', 'vomiting', 'nutrition', 'feeding', 'bowel']
        }
    
    def generate_medical_embeddings(self, force_regenerate: bool = False, sample_size: Optional[int] = None):
        """Generate medical-optimized embeddings"""
        
        if os.path.exists(self.embedding_file) and not force_regenerate:
            print(f"Loading existing medical embeddings...")
            with open(self.embedding_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"✅ Loaded embeddings: {self.embeddings.shape}")
            return
        
        print("🧬 Generating medical embeddings...")
        
        # Sample for demo if requested
        if sample_size and len(self.df) > sample_size:
            print(f"Using sample of {sample_size} chunks for demonstration")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Prepare medical-enhanced texts
        texts = []
        for _, row in self.df.iterrows():
            # Combine multiple fields for richer context
            medical_text = (
                f"Medical Topic: {row['chapter_title']} "
                f"Clinical Section: {row['section_title']} "
                f"Summary: {row['chunk_summary']} "
                f"Keywords: {row['keywords']} "
                f"Content: {row['chunk_text']}"
            )
            texts.append(medical_text)
        
        # Fit and encode
        print("Training medical embedding model...")
        self.embedding_model.fit(texts)
        
        print("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(texts)
        
        print(f"✅ Generated medical embeddings: {self.embeddings.shape}")
        
        # Save embeddings
        with open(self.embedding_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"💾 Saved embeddings to {self.embedding_file}")
    
    def medical_search(self, query: str, specialty_context: Optional[str] = None, 
                      top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Perform medical-aware semantic search"""
        
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call generate_medical_embeddings() first.")
        
        # Enhance query with medical context
        enhanced_query = self._enhance_query_with_medical_context(query, specialty_context)
        
        # Encode enhanced query
        query_embedding = self.embedding_model.encode([enhanced_query])[0]
        
        # Calculate medical similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= min_similarity:
                row = self.df.iloc[idx]
                
                # Calculate medical relevance score
                medical_relevance = self._calculate_medical_relevance(query, row)
                
                result = {
                    'chunk_id': str(row['id']),
                    'similarity': similarity,
                    'medical_relevance': medical_relevance,
                    'combined_score': (similarity * 0.7) + (medical_relevance * 0.3),
                    'page_number': int(row['page_number']),
                    'chapter_title': str(row['chapter_title']),
                    'section_path': str(row['section_heading_path']),
                    'summary': str(row['chunk_summary']),
                    'keywords': str(row['keywords']),
                    'text_preview': str(row['chunk_text'])[:400] + "...",
                    'full_text': str(row['chunk_text']),
                    'confidence': float(row['confidence_score']),
                    'citation': self._format_citation(row),
                    'detected_specialty': self._detect_specialty(row['chunk_text'])
                }
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results
    
    def _enhance_query_with_medical_context(self, query: str, specialty: Optional[str] = None) -> str:
        """Enhance query with medical context"""
        enhanced = f"pediatric medical {query}"
        
        if specialty and specialty in self.medical_specialties:
            context_terms = " ".join(self.medical_specialties[specialty])
            enhanced += f" {context_terms}"
        
        return enhanced
    
    def _calculate_medical_relevance(self, query: str, row: pd.Series) -> float:
        """Calculate medical relevance score"""
        score = 0.0
        query_lower = query.lower()
        text_lower = str(row['chunk_text']).lower()
        keywords_lower = str(row['keywords']).lower()
        
        # Direct term matches
        for term in query_lower.split():
            if term in text_lower:
                score += 0.1
            if term in keywords_lower:
                score += 0.2
        
        # Medical term bonuses
        medical_terms_in_query = [term for term in self.embedding_model.medical_terms 
                                 if term in query_lower]
        medical_terms_in_text = [term for term in self.embedding_model.medical_terms 
                                if term in text_lower]
        
        common_medical_terms = set(medical_terms_in_query) & set(medical_terms_in_text)
        score += len(common_medical_terms) * 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _detect_specialty(self, text: str) -> str:
        """Detect medical specialty from text"""
        text_lower = text.lower()
        specialty_scores = {}
        
        for specialty, terms in self.medical_specialties.items():
            score = sum(1 for term in terms if term in text_lower)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            return max(specialty_scores, key=specialty_scores.get)
        return "general"
    
    def _format_citation(self, row: pd.Series) -> str:
        """Format medical citation"""
        return (f"{row['authors']} ({row['year']}). {row['book_title']}, "
               f"{row['edition']} Ed. {row['publisher']}. "
               f"Chapter: {row['chapter_title']}, p. {row['page_number']}.")
    
    def clinical_query_demo(self):
        """Demonstrate clinical queries with the medical RAG system"""
        clinical_scenarios = [
            {
                'query': 'fever in 3-month-old infant management',
                'specialty': 'infectious_disease',
                'description': 'Young infant fever evaluation'
            },
            {
                'query': 'type 1 diabetes insulin therapy children',
                'specialty': 'endocrinology', 
                'description': 'Pediatric diabetes management'
            },
            {
                'query': 'asthma exacerbation treatment pediatric emergency',
                'specialty': 'pulmonology',
                'description': 'Acute asthma management'
            }
        ]
        
        print("\n🏥 CLINICAL QUERY DEMONSTRATION")
        print("=" * 70)
        
        for scenario in clinical_scenarios:
            print(f"\n📋 Clinical Scenario: {scenario['description']}")
            print(f"🔍 Query: {scenario['query']}")
            print(f"🏥 Specialty Context: {scenario['specialty']}")
            print("-" * 50)
            
            results = self.medical_search(
                scenario['query'], 
                specialty_context=scenario['specialty'],
                top_k=2
            )
            
            for i, result in enumerate(results, 1):
                print(f"\n🔸 Result {i}")
                print(f"   📊 Similarity: {result['similarity']:.3f}")
                print(f"   🏥 Medical Relevance: {result['medical_relevance']:.3f}")
                print(f"   🎯 Combined Score: {result['combined_score']:.3f}")
                print(f"   📍 Page {result['page_number']} | {result['chapter_title']}")
                print(f"   🏷️  Detected Specialty: {result['detected_specialty']}")
                print(f"   📝 {result['summary']}")
                print(f"   🔗 {result['citation'][:80]}...")

def run_advanced_medical_rag_demo():
    """Run comprehensive demo of the advanced medical RAG system"""
    
    print("🏥 ADVANCED MEDICAL RAG SYSTEM")
    print("=" * 70)
    print("Optimized for pediatric medical content with:")
    print("✅ Medical vocabulary enhancement")
    print("✅ Specialty-aware search") 
    print("✅ Clinical relevance scoring")
    print("✅ Medical term pattern recognition")
    print("=" * 70)
    
    # Initialize system
    csv_path = "/project/workspace/nelson_chunks.csv"
    rag = AdvancedMedicalRAG(csv_path, embedding_dim=256)  # Smaller for demo
    
    # Generate embeddings (sample for demo)
    rag.generate_medical_embeddings(sample_size=500)
    
    # Run clinical demonstrations
    rag.clinical_query_demo()
    
    print(f"\n" + "="*70)
    print("🚀 PRODUCTION DEPLOYMENT")
    print("="*70)
    print("To deploy for full medical knowledge base:")
    print("1. Run with full dataset: generate_medical_embeddings(sample_size=None)")
    print("2. Increase embedding dimensions for better accuracy")
    print("3. Fine-tune medical term weights for your use case")
    print("4. Add specialty-specific models for targeted searches")
    
    return rag

if __name__ == "__main__":
    system = run_advanced_medical_rag_demo()