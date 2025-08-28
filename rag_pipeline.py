#!/usr/bin/env python3
"""
RAG Pipeline for Nelson Textbook of Pediatrics
Creates a comprehensive knowledge base with rich metadata for citation and tracing
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

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

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

class PediatricsRAGPipeline:
    def __init__(self, pdf_path: str, drive_link: str):
        """Initialize the RAG pipeline"""
        self.pdf_path = pdf_path
        self.drive_link = drive_link
        
        # Book metadata (hardcoded for Nelson Textbook of Pediatrics 22e)
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
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Chapter and section patterns
        self.chapter_patterns = [
            r'^CHAPTER\s+(\d+)\s*[:\-]?\s*(.+)$',
            r'^Chapter\s+(\d+)\s*[:\-]?\s*(.+)$',
            r'^(\d+)\.\s+(.+)$',
        ]
        
        self.section_patterns = [
            r'^([A-Z][A-Z\s&]+)$',  # All caps sections
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',  # Title case sections
            r'^\d+\.\d+\s+(.+)$',  # Numbered subsections
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def extract_pdf_content(self) -> List[Dict]:
        """Extract text content from PDF with page and structure information"""
        print("Extracting PDF content...")
        doc = fitz.open(self.pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract text blocks with font information for structure detection
            blocks = page.get_text("dict")["blocks"]
            
            page_data = {
                "page_number": page_num + 1,
                "text": text,
                "blocks": blocks,
                "lines": text.split('\n') if text else []
            }
            pages.append(page_data)
            
            if page_num % 100 == 0:
                print(f"Processed page {page_num + 1}/{len(doc)}")
        
        doc.close()
        print(f"Extracted content from {len(pages)} pages")
        return pages

    def detect_structure(self, pages: List[Dict]) -> List[Dict]:
        """Detect chapters, sections, and headings in the document"""
        print("Detecting document structure...")
        structured_pages = []
        current_chapter = {"number": "", "title": "Introduction"}
        current_section = "General"
        current_subsection = ""
        
        for page_data in pages:
            page_structure = {
                **page_data,
                "chapter": current_chapter.copy(),
                "section": current_section,
                "subsection": current_subsection,
                "headings": []
            }
            
            for line in page_data["lines"]:
                line = line.strip()
                if not line:
                    continue
                
                # Check for chapter patterns
                chapter_match = None
                for pattern in self.chapter_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        chapter_match = match
                        break
                
                if chapter_match:
                    current_chapter = {
                        "number": chapter_match.group(1) if len(chapter_match.groups()) > 1 else "",
                        "title": chapter_match.group(2) if len(chapter_match.groups()) > 1 else chapter_match.group(1)
                    }
                    current_section = "General"
                    current_subsection = ""
                    page_structure["headings"].append(("chapter", line))
                    continue
                
                # Check for section patterns
                section_match = None
                for pattern in self.section_patterns:
                    match = re.match(pattern, line)
                    if match and len(line) < 100:  # Headings are typically shorter
                        section_match = match
                        break
                
                if section_match and len(line.split()) <= 10:  # Max 10 words for headings
                    if line.isupper() or (len(line.split()) <= 5 and any(word[0].isupper() for word in line.split())):
                        current_section = line
                        current_subsection = ""
                        page_structure["headings"].append(("section", line))
                    elif re.match(r'^\d+\.\d+', line):
                        current_subsection = line
                        page_structure["headings"].append(("subsection", line))
            
            page_structure["chapter"] = current_chapter.copy()
            page_structure["section"] = current_section
            page_structure["subsection"] = current_subsection
            structured_pages.append(page_structure)
        
        print("Document structure detection complete")
        return structured_pages

    def create_chunks(self, structured_pages: List[Dict], target_tokens: int = 500, overlap_tokens: int = 75) -> List[Dict]:
        """Create overlapping chunks from structured pages"""
        print(f"Creating chunks with target size {target_tokens} tokens and {overlap_tokens} token overlap...")
        chunks = []
        
        for page_data in structured_pages:
            if not page_data["text"].strip():
                continue
            
            # Split page text into paragraphs
            paragraphs = [p.strip() for p in page_data["text"].split('\n\n') if p.strip()]
            
            current_chunk = ""
            chunk_paragraphs = []
            
            for paragraph in paragraphs:
                # Calculate tokens if we add this paragraph
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                token_count = self.count_tokens(test_chunk)
                
                if token_count <= target_tokens + 100:  # Allow slight overflow
                    current_chunk = test_chunk
                    chunk_paragraphs.append(paragraph)
                else:
                    # Finalize current chunk if it has content
                    if current_chunk:
                        chunk_data = self.create_chunk_metadata(
                            current_chunk, page_data, len(chunks)
                        )
                        chunks.append(chunk_data)
                    
                    # Start new chunk with overlap from previous chunk
                    overlap_text = ""
                    if chunk_paragraphs:
                        # Take last paragraph(s) for overlap
                        overlap_paragraphs = chunk_paragraphs[-1:]
                        overlap_text = "\n\n".join(overlap_paragraphs)
                        overlap_token_count = self.count_tokens(overlap_text)
                        
                        # If overlap is too small, take more paragraphs
                        if overlap_token_count < overlap_tokens and len(chunk_paragraphs) > 1:
                            overlap_paragraphs = chunk_paragraphs[-2:]
                            overlap_text = "\n\n".join(overlap_paragraphs)
                    
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                    chunk_paragraphs = overlap_paragraphs + [paragraph] if overlap_text else [paragraph]
            
            # Add final chunk if it has content
            if current_chunk:
                chunk_data = self.create_chunk_metadata(
                    current_chunk, page_data, len(chunks)
                )
                chunks.append(chunk_data)
        
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_chunk_metadata(self, chunk_text: str, page_data: Dict, chunk_index: int) -> ChunkMetadata:
        """Create metadata for a text chunk"""
        chunk_id = str(uuid.uuid4())
        
        # Build hierarchical path
        path_parts = []
        if page_data["chapter"]["title"]:
            chapter_part = f"Chapter {page_data['chapter']['number']}: {page_data['chapter']['title']}" if page_data['chapter']['number'] else page_data['chapter']['title']
            path_parts.append(chapter_part)
        
        if page_data["section"] and page_data["section"] != "General":
            path_parts.append(page_data["section"])
        
        if page_data["subsection"]:
            path_parts.append(page_data["subsection"])
        
        section_heading_path = " > ".join(path_parts) if path_parts else "Introduction"
        
        # Generate summary and keywords
        summary = self.generate_summary(chunk_text)
        keywords = self.extract_keywords(chunk_text)
        
        # Calculate confidence score based on text quality
        confidence_score = self.calculate_confidence_score(chunk_text, page_data)
        
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
            chunk_summary=summary,
            keywords=keywords,
            confidence_score=confidence_score,
            created_at=datetime.now().isoformat()
        )

    def generate_summary(self, text: str) -> str:
        """Generate a 1-2 sentence summary of the chunk"""
        # Simple extractive summarization - take first 1-2 sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not sentences:
            return "Medical text content."
        
        # Return first sentence or two, up to 200 characters
        summary = sentences[0]
        if len(summary) < 100 and len(sentences) > 1:
            summary += ". " + sentences[1]
        
        # Truncate if too long
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        return summary

    def extract_keywords(self, text: str, max_keywords: int = 10) -> str:
        """Extract key medical terms and concepts from the text"""
        # Clean and tokenize text
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        # Medical term patterns
        medical_patterns = [
            r'\b\w*itis\b',  # Inflammations
            r'\b\w*osis\b',  # Conditions
            r'\b\w*emia\b',  # Blood conditions
            r'\b\w*pathy\b', # Diseases
            r'\b\w*therapy\b', # Treatments
            r'\b\w*syndrome\b', # Syndromes
        ]
        
        keywords = set()
        
        # Extract medical terms using patterns
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.update([m.lower() for m in matches if len(m) > 4])
        
        # Extract other important terms (capitalized words, numbers with units)
        important_terms = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\s*(?:mg|kg|ml|years?|months?|days?)\b', text)
        keywords.update([t.lower() for t in important_terms if len(t) > 2])
        
        # Use TF-IDF for additional keyword extraction if we have enough text
        if len(text) > 500:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=max_keywords,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top scoring terms
                top_indices = tfidf_scores.argsort()[-max_keywords:][::-1]
                tfidf_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                keywords.update(tfidf_keywords)
            except:
                pass
        
        # Filter and limit keywords
        filtered_keywords = [
            kw for kw in keywords 
            if len(kw) > 2 and len(kw) < 30 and not kw.isdigit()
        ]
        
        return ",".join(list(filtered_keywords)[:max_keywords])

    def calculate_confidence_score(self, text: str, page_data: Dict) -> float:
        """Calculate confidence score for chunk quality (0-1)"""
        score = 0.5  # Base score
        
        # Length factor
        if 200 <= len(text) <= 2000:
            score += 0.2
        elif len(text) > 100:
            score += 0.1
        
        # Structure factor
        if page_data["chapter"]["title"] and page_data["chapter"]["title"] != "Introduction":
            score += 0.1
        if page_data["section"] and page_data["section"] != "General":
            score += 0.1
        
        # Content quality indicators
        medical_terms = len(re.findall(r'\b(?:patient|treatment|diagnosis|syndrome|disease|therapy|clinical|medical)\b', text.lower()))
        if medical_terms > 0:
            score += min(0.2, medical_terms * 0.02)
        
        # Penalize very short or very long chunks
        if len(text) < 100:
            score -= 0.2
        elif len(text) > 3000:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def export_to_csv(self, chunks: List[ChunkMetadata], output_file: str):
        """Export chunks to CSV format"""
        print(f"Exporting {len(chunks)} chunks to {output_file}")
        
        # Convert to list of dictionaries
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
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, quoting=1)  # Quote all fields
        
        print(f"Successfully exported to {output_file}")
        
        # Print statistics
        total_tokens = sum(chunk.chunk_token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        avg_confidence = sum(chunk.confidence_score for chunk in chunks) / len(chunks) if chunks else 0
        
        print(f"\nStatistics:")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average tokens per chunk: {avg_tokens:.1f}")
        print(f"Average confidence score: {avg_confidence:.3f}")
        
        # Page coverage
        pages_covered = len(set(chunk.page_number for chunk in chunks))
        print(f"Pages covered: {pages_covered}")

    def run_pipeline(self, output_file: str = "nelson_chunks.csv"):
        """Run the complete RAG pipeline"""
        print("Starting Nelson Textbook of Pediatrics RAG Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Extract PDF content
            pages = self.extract_pdf_content()
            
            # Step 2: Detect structure
            structured_pages = self.detect_structure(pages)
            
            # Step 3: Create chunks
            chunks = self.create_chunks(structured_pages)
            
            # Step 4: Export to CSV
            self.export_to_csv(chunks, output_file)
            
            print("=" * 60)
            print("RAG Pipeline completed successfully!")
            
            return chunks
            
        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            raise

def main():
    """Main function to run the RAG pipeline"""
    pdf_path = "/project/workspace/nelson_pediatrics_textbook_22e.pdf"
    drive_link = "https://drive.google.com/file/d/1KvjRFW_x-qdXj774UjyO388Ve5lffXgg/view?usp=drivesdk"
    output_file = "/project/workspace/nelson_chunks.csv"
    
    pipeline = PediatricsRAGPipeline(pdf_path, drive_link)
    chunks = pipeline.run_pipeline(output_file)
    
    return chunks

if __name__ == "__main__":
    main()