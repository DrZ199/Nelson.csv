#!/usr/bin/env python3
"""
Improved RAG Pipeline with Better Chunking Strategy
Optimized for 500-token chunks with semantic boundaries
"""

import fitz  # PyMuPDF
import pandas as pd
import re
import uuid
import nltk
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
import math

@dataclass
class ChunkMetadata:
    """Metadata for each text chunk"""
    id: str
    book_title: str
    edition: str
    authors: str
    publisher: str
    year: str
    isbn: str
    source_url: str
    drive_link: str
    page_number: int
    chapter_title: str
    section_title: str
    section_heading_path: str
    chunk_index: int
    chunk_token_count: int
    chunk_text: str
    chunk_summary: str
    keywords: str
    confidence_score: float
    created_at: str

class ImprovedPediatricsRAG:
    def __init__(self, pdf_path: str, drive_link: str):
        """Initialize improved RAG pipeline"""
        self.pdf_path = pdf_path
        self.drive_link = drive_link
        
        # Book metadata
        self.book_metadata = {
            "book_title": "Nelson Textbook of Pediatrics",
            "edition": "22nd",
            "authors": "Robert M. Kliegman; Bonita Stanton; Joseph St. Geme; Nina F. Schor; Richard E. Behrman",
            "publisher": "Elsevier",
            "year": "2019",
            "isbn": "978-0323529501",
            "source_url": "https://www.elsevier.com/books/nelson-textbook-of-pediatrics/kliegman/978-0-323-52950-1",
            "drive_link": drive_link
        }
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Improved patterns for medical text
        self.chapter_patterns = [
            r'^CHAPTER\s+(\d+)\s*[:\-•]?\s*(.+)$',
            r'^Chapter\s+(\d+)\s*[:\-•]?\s*(.+)$',
            r'^(\d+)\.\s+(.+)$',
            r'^Part\s+([IVXLC]+)\s*[:\-•]?\s*(.+)$',
        ]
        
        self.section_patterns = [
            r'^([A-Z][A-Z\s&]{3,})$',  # All caps sections
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',  # Title case
            r'^\d+\.\d+\.?\s+(.+)$',  # Numbered subsections
            r'^([A-Z][a-z]+.*?)\s*$',  # Medical headings
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences with medical text awareness"""
        # Handle medical abbreviations and numbers
        text = re.sub(r'\b([A-Z]{2,})\.\s', r'\1<ABBREV> ', text)  # Protect abbreviations
        text = re.sub(r'\b(\d+)\.\s*(\d+)', r'\1<DOT>\2', text)  # Protect decimal numbers
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore protected elements
        sentences = [s.replace('<ABBREV>', '.').replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]

    def create_semantic_chunks(self, text: str, page_data: Dict, 
                             target_tokens: int = 500, 
                             overlap_tokens: int = 75) -> List[str]:
        """Create semantically meaningful chunks"""
        sentences = self.extract_sentences(text)
        chunks = []
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed target, finalize current chunk
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Create overlap by keeping last few sentences
                overlap_sentences = []
                overlap_tokens_count = 0
                
                for prev_sentence in reversed(current_chunk):
                    prev_tokens = self.count_tokens(prev_sentence)
                    if overlap_tokens_count + prev_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_tokens_count += prev_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

    def process_improved_chunks(self) -> List[ChunkMetadata]:
        """Process PDF with improved chunking strategy"""
        print("Starting improved chunking process...")
        
        # Extract PDF content (simplified for demo)
        doc = fitz.open(self.pdf_path)
        all_chunks = []
        
        # Process first 100 pages as demo (full processing would take longer)
        sample_pages = min(100, len(doc))
        print(f"Processing sample of {sample_pages} pages...")
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Simple structure detection for demo
            page_data = {
                "page_number": page_num + 1,
                "chapter": {"title": f"Chapter {page_num // 10 + 1}", "number": str(page_num // 10 + 1)},
                "section": "General",
                "subsection": ""
            }
            
            # Create semantic chunks
            chunk_texts = self.create_semantic_chunks(text, page_data)
            
            for i, chunk_text in enumerate(chunk_texts):
                chunk_metadata = self.create_chunk_metadata(
                    chunk_text, page_data, len(all_chunks)
                )
                all_chunks.append(chunk_metadata)
            
            if page_num % 20 == 0:
                print(f"Processed page {page_num + 1}/{sample_pages}")
        
        doc.close()
        print(f"Created {len(all_chunks)} improved chunks")
        return all_chunks

    def create_chunk_metadata(self, chunk_text: str, page_data: Dict, chunk_index: int) -> ChunkMetadata:
        """Create metadata for chunk"""
        chunk_id = str(uuid.uuid4())
        
        # Build path
        path_parts = []
        if page_data["chapter"]["title"]:
            path_parts.append(page_data["chapter"]["title"])
        if page_data["section"]:
            path_parts.append(page_data["section"])
        if page_data["subsection"]:
            path_parts.append(page_data["subsection"])
        
        section_heading_path = " > ".join(path_parts) if path_parts else "Introduction"
        
        return ChunkMetadata(
            id=chunk_id,
            book_title=self.book_metadata["book_title"],
            edition=self.book_metadata["edition"],
            authors=self.book_metadata["authors"],
            publisher=self.book_metadata["publisher"],
            year=self.book_metadata["year"],
            isbn=self.book_metadata["isbn"],
            source_url=self.book_metadata["source_url"],
            drive_link=self.book_metadata["drive_link"],
            page_number=page_data["page_number"],
            chapter_title=page_data["chapter"]["title"],
            section_title=page_data["section"],
            section_heading_path=section_heading_path,
            chunk_index=chunk_index,
            chunk_token_count=self.count_tokens(chunk_text),
            chunk_text=chunk_text,
            chunk_summary=self.generate_summary(chunk_text),
            keywords=self.extract_keywords(chunk_text),
            confidence_score=0.9,  # Simplified for demo
            created_at=datetime.now().isoformat()
        )

    def generate_summary(self, text: str) -> str:
        """Generate summary"""
        sentences = self.extract_sentences(text)
        if not sentences:
            return "Medical content."
        
        summary = sentences[0]
        if len(summary) < 100 and len(sentences) > 1:
            summary += " " + sentences[1]
        
        return summary[:200] + "..." if len(summary) > 200 else summary

    def extract_keywords(self, text: str) -> str:
        """Extract medical keywords"""
        # Medical patterns
        medical_terms = re.findall(
            r'\b(?:syndrome|disease|disorder|condition|treatment|therapy|diagnosis|symptom|medication|drug)\b',
            text.lower()
        )
        
        # Capitalized terms (likely medical names)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Numbers with units
        measurements = re.findall(r'\d+\s*(?:mg|ml|kg|years?|months?|days?|%)', text)
        
        all_keywords = list(set(medical_terms + capitalized[:5] + measurements))
        return ",".join(all_keywords[:10])

def run_improved_demo():
    """Run improved chunking demo"""
    pdf_path = "/project/workspace/nelson_pediatrics_textbook_22e.pdf"
    drive_link = "https://drive.google.com/file/d/1KvjRFW_x-qdXj774UjyO388Ve5lffXgg/view?usp=drivesdk"
    
    improved_rag = ImprovedPediatricsRAG(pdf_path, drive_link)
    chunks = improved_rag.process_improved_chunks()
    
    # Convert to DataFrame and save sample
    data = []
    for chunk in chunks:
        data.append({
            "id": chunk.id,
            "book_title": chunk.book_title,
            "edition": chunk.edition,
            "authors": chunk.authors,
            "publisher": chunk.publisher,
            "year": chunk.year,
            "isbn": chunk.isbn,
            "source_url": chunk.source_url,
            "drive_link": chunk.drive_link,
            "page_number": chunk.page_number,
            "chapter_title": chunk.chapter_title,
            "section_title": chunk.section_title,
            "section_heading_path": chunk.section_heading_path,
            "chunk_index": chunk.chunk_index,
            "chunk_token_count": chunk.chunk_token_count,
            "chunk_text": chunk.chunk_text,
            "chunk_summary": chunk.chunk_summary,
            "keywords": chunk.keywords,
            "confidence_score": chunk.confidence_score,
            "created_at": chunk.created_at
        })
    
    df = pd.DataFrame(data)
    df.to_csv("/project/workspace/nelson_chunks_improved_sample.csv", index=False, quoting=1)
    
    # Statistics
    total_tokens = sum(chunk.chunk_token_count for chunk in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    
    print(f"\nImproved Chunking Results:")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    print(f"Token range: {min(c.chunk_token_count for c in chunks)} - {max(c.chunk_token_count for c in chunks)}")
    
    # Show examples
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} (Page {chunk.page_number}, {chunk.chunk_token_count} tokens) ---")
        print(f"Summary: {chunk.chunk_summary}")
        print(f"Keywords: {chunk.keywords}")
        print(f"Text: {chunk.chunk_text[:200]}...")

if __name__ == "__main__":
    run_improved_demo()