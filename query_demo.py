#!/usr/bin/env python3
"""
Query System for Nelson Pediatrics RAG Knowledge Base
Demonstrates how to search and retrieve information with citations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple
import json

class PediatricsQuerySystem:
    def __init__(self, csv_path: str):
        """Initialize the query system with the chunks CSV"""
        print("Loading knowledge base...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} chunks from knowledge base")
        
        # Initialize TF-IDF vectorizer
        print("Building search index...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Fit vectorizer on chunk texts
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['chunk_text'].fillna(''))
        print("Search index built successfully")

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Search for relevant chunks based on query"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top matches above minimum similarity
        top_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in top_indices[:top_k]:
            if similarities[idx] >= min_similarity:
                row = self.df.iloc[idx]
                result = {
                    'chunk_id': row['id'],
                    'similarity': float(similarities[idx]),
                    'page_number': int(row['page_number']),
                    'chapter_title': str(row['chapter_title']),
                    'section_path': str(row['section_heading_path']),
                    'text': str(row['chunk_text'])[:500] + "..." if len(str(row['chunk_text'])) > 500 else str(row['chunk_text']),
                    'full_text': str(row['chunk_text']),
                    'summary': str(row['chunk_summary']),
                    'keywords': str(row['keywords']),
                    'confidence_score': float(row['confidence_score']),
                    'citation': self.format_citation(row)
                }
                results.append(result)
        
        return results

    def format_citation(self, row) -> str:
        """Format a proper citation for the chunk"""
        citation = f"{row['authors']}. {row['book_title']}, {row['edition']} Edition. "
        citation += f"{row['publisher']}, {row['year']}. "
        citation += f"Page {row['page_number']}. "
        citation += f"Chapter: {row['chapter_title']}. "
        citation += f"Section: {row['section_heading_path']}."
        return citation

    def answer_question(self, question: str, max_context: int = 2000) -> Dict:
        """Answer a question using the knowledge base"""
        print(f"\nSearching for: '{question}'")
        
        # Search for relevant chunks
        results = self.search(question, top_k=3)
        
        if not results:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the knowledge base.",
                'sources': []
            }
        
        # Combine relevant context
        context_parts = []
        sources = []
        current_length = 0
        
        for result in results:
            if current_length + len(result['full_text']) <= max_context:
                context_parts.append(result['full_text'])
                current_length += len(result['full_text'])
                sources.append({
                    'page': result['page_number'],
                    'chapter': result['chapter_title'],
                    'section': result['section_path'],
                    'similarity': result['similarity'],
                    'citation': result['citation']
                })
        
        combined_context = "\n\n---\n\n".join(context_parts)
        
        # For this demo, we'll return the most relevant excerpts
        # In a production system, you'd use an LLM to generate answers
        
        return {
            'question': question,
            'relevant_excerpts': [result['text'] for result in results],
            'sources': sources,
            'raw_context': combined_context
        }

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base"""
        stats = {
            'total_chunks': len(self.df),
            'total_pages': self.df['page_number'].max(),
            'total_tokens': int(self.df['chunk_token_count'].sum()),
            'avg_tokens_per_chunk': float(self.df['chunk_token_count'].mean()),
            'avg_confidence': float(self.df['confidence_score'].mean()),
            'unique_chapters': self.df['chapter_title'].nunique(),
            'chapter_distribution': self.df['chapter_title'].value_counts().head(10).to_dict()
        }
        return stats

    def browse_by_chapter(self, chapter_query: str, limit: int = 10) -> List[Dict]:
        """Browse chunks by chapter name"""
        # Find chapters matching the query
        matching_chapters = self.df[
            self.df['chapter_title'].str.contains(chapter_query, case=False, na=False)
        ].head(limit)
        
        results = []
        for _, row in matching_chapters.iterrows():
            results.append({
                'page_number': int(row['page_number']),
                'chapter_title': str(row['chapter_title']),
                'section_path': str(row['section_heading_path']),
                'summary': str(row['chunk_summary']),
                'keywords': str(row['keywords']),
                'token_count': int(row['chunk_token_count']),
                'confidence': float(row['confidence_score'])
            })
        
        return results

def demo_queries():
    """Run demonstration queries"""
    csv_path = "/project/workspace/nelson_chunks.csv"
    query_system = PediatricsQuerySystem(csv_path)
    
    # Show statistics
    print("\n" + "="*60)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*60)
    stats = query_system.get_statistics()
    for key, value in stats.items():
        if key != 'chapter_distribution':
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\nTop Chapters by Chunk Count:")
    for chapter, count in list(stats['chapter_distribution'].items())[:5]:
        print(f"  - {chapter}: {count} chunks")
    
    # Sample queries
    queries = [
        "diabetes mellitus type 1 treatment",
        "fever in infants",
        "asthma management",
        "vaccination schedule",
        "congenital heart disease"
    ]
    
    print("\n" + "="*60)
    print("SAMPLE QUERIES AND RESULTS")
    print("="*60)
    
    for query in queries:
        print(f"\n🔍 QUERY: {query}")
        print("-" * 50)
        
        results = query_system.search(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n📄 Result {i} (Similarity: {result['similarity']:.3f})")
                print(f"📍 Location: Page {result['page_number']} | {result['section_path']}")
                print(f"📝 Summary: {result['summary']}")
                print(f"🏷️ Keywords: {result['keywords']}")
                print(f"📖 Citation: {result['citation'][:100]}...")
                print(f"📄 Text Preview: {result['text'][:200]}...")
        else:
            print("No relevant results found.")
    
    # Chapter browsing example
    print(f"\n" + "="*60)
    print("CHAPTER BROWSING EXAMPLE: 'Cardiology'")
    print("="*60)
    
    cardio_chunks = query_system.browse_by_chapter("cardio", limit=3)
    for i, chunk in enumerate(cardio_chunks, 1):
        print(f"\n📄 Chunk {i}")
        print(f"📍 Page {chunk['page_number']} | {chunk['section_path']}")
        print(f"📝 {chunk['summary']}")
        print(f"🏷️ {chunk['keywords'][:100]}...")

if __name__ == "__main__":
    demo_queries()