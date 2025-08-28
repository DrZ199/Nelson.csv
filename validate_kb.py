#!/usr/bin/env python3
"""
Knowledge Base Validation and Testing
Validates the quality and completeness of the RAG pipeline output
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import json
from typing import Dict, List

class KnowledgeBaseValidator:
    def __init__(self, csv_path: str):
        """Initialize validator with CSV path"""
        print("Loading knowledge base for validation...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} chunks for validation")

    def validate_completeness(self) -> Dict:
        """Validate data completeness"""
        print("\n=== COMPLETENESS VALIDATION ===")
        
        completeness = {
            'total_chunks': len(self.df),
            'missing_data': {},
            'page_coverage': {},
            'token_distribution': {}
        }
        
        # Check for missing required fields
        required_fields = [
            'id', 'chunk_text', 'page_number', 'chunk_token_count',
            'chapter_title', 'section_heading_path'
        ]
        
        for field in required_fields:
            missing_count = self.df[field].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            completeness['missing_data'][field] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2)
            }
            print(f"  {field}: {missing_count} missing ({missing_pct:.1f}%)")
        
        # Page coverage analysis
        min_page = self.df['page_number'].min()
        max_page = self.df['page_number'].max()
        unique_pages = self.df['page_number'].nunique()
        expected_pages = max_page - min_page + 1
        
        completeness['page_coverage'] = {
            'min_page': int(min_page),
            'max_page': int(max_page),
            'unique_pages': int(unique_pages),
            'expected_pages': int(expected_pages),
            'coverage_percentage': round((unique_pages / expected_pages) * 100, 2)
        }
        
        print(f"  Page coverage: {unique_pages}/{expected_pages} pages ({completeness['page_coverage']['coverage_percentage']:.1f}%)")
        
        return completeness

    def validate_token_distribution(self) -> Dict:
        """Analyze token count distribution"""
        print("\n=== TOKEN DISTRIBUTION ANALYSIS ===")
        
        tokens = self.df['chunk_token_count'].dropna()
        
        distribution = {
            'mean': float(tokens.mean()),
            'median': float(tokens.median()),
            'std': float(tokens.std()),
            'min': int(tokens.min()),
            'max': int(tokens.max()),
            'percentiles': {
                '25th': float(tokens.quantile(0.25)),
                '75th': float(tokens.quantile(0.75)),
                '95th': float(tokens.quantile(0.95))
            }
        }
        
        print(f"  Mean tokens per chunk: {distribution['mean']:.1f}")
        print(f"  Median tokens: {distribution['median']:.1f}")
        print(f"  Token range: {distribution['min']} - {distribution['max']}")
        print(f"  Target was ~500 tokens")
        
        # Categorize chunks by size
        size_categories = {
            'very_small': len(tokens[tokens < 200]),
            'small': len(tokens[(tokens >= 200) & (tokens < 400)]),
            'target': len(tokens[(tokens >= 400) & (tokens < 700)]),
            'large': len(tokens[(tokens >= 700) & (tokens < 1000)]),
            'very_large': len(tokens[tokens >= 1000])
        }
        
        print(f"\n  Size distribution:")
        for category, count in size_categories.items():
            pct = (count / len(tokens)) * 100
            print(f"    {category.replace('_', ' ').title()}: {count} chunks ({pct:.1f}%)")
        
        return distribution

    def validate_content_quality(self) -> Dict:
        """Validate content quality"""
        print("\n=== CONTENT QUALITY VALIDATION ===")
        
        quality = {
            'confidence_scores': {},
            'medical_content': {},
            'text_characteristics': {}
        }
        
        # Confidence score analysis
        confidence_scores = self.df['confidence_score'].dropna()
        quality['confidence_scores'] = {
            'mean': float(confidence_scores.mean()),
            'median': float(confidence_scores.median()),
            'std': float(confidence_scores.std()),
            'low_confidence_count': int(len(confidence_scores[confidence_scores < 0.5]))
        }
        
        print(f"  Average confidence score: {quality['confidence_scores']['mean']:.3f}")
        print(f"  Low confidence chunks (<0.5): {quality['confidence_scores']['low_confidence_count']}")
        
        # Medical content analysis
        sample_texts = self.df['chunk_text'].dropna().sample(min(1000, len(self.df)))
        
        medical_indicators = [
            (r'\bpatient\b', 'patient_mentions'),
            (r'\btreatment\b', 'treatment_mentions'),
            (r'\bdiagnosis\b', 'diagnosis_mentions'),
            (r'\bsyndrome\b', 'syndrome_mentions'),
            (r'\bdisease\b', 'disease_mentions'),
            (r'\btherapy\b', 'therapy_mentions'),
        ]
        
        for pattern, name in medical_indicators:
            count = sum(1 for text in sample_texts if re.search(pattern, text, re.IGNORECASE))
            percentage = (count / len(sample_texts)) * 100
            quality['medical_content'][name] = {
                'count': count,
                'percentage': round(percentage, 1)
            }
            print(f"  Chunks with '{name.replace('_', ' ')}': {count}/{len(sample_texts)} ({percentage:.1f}%)")
        
        return quality

    def validate_citations(self) -> Dict:
        """Validate citation formatting"""
        print("\n=== CITATION VALIDATION ===")
        
        citation_issues = {
            'missing_authors': 0,
            'missing_isbn': 0,
            'invalid_page_numbers': 0,
            'missing_chapters': 0
        }
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['authors']) or not row['authors']:
                citation_issues['missing_authors'] += 1
            
            if pd.isna(row['isbn']) or not row['isbn']:
                citation_issues['missing_isbn'] += 1
            
            if pd.isna(row['page_number']) or row['page_number'] <= 0:
                citation_issues['invalid_page_numbers'] += 1
            
            if pd.isna(row['chapter_title']) or not row['chapter_title']:
                citation_issues['missing_chapters'] += 1
        
        total_chunks = len(self.df)
        for issue, count in citation_issues.items():
            percentage = (count / total_chunks) * 100
            print(f"  {issue.replace('_', ' ').title()}: {count} chunks ({percentage:.1f}%)")
        
        return citation_issues

    def generate_sample_queries(self) -> List[Dict]:
        """Generate sample queries from the knowledge base"""
        print("\n=== SAMPLE QUERY GENERATION ===")
        
        # Extract common medical terms from keywords
        all_keywords = []
        for keywords in self.df['keywords'].dropna():
            all_keywords.extend(keywords.split(','))
        
        # Clean and count keywords
        clean_keywords = [kw.strip().lower() for kw in all_keywords if kw.strip()]
        keyword_counts = Counter(clean_keywords)
        
        # Get top medical terms
        common_medical_terms = [
            term for term, count in keyword_counts.most_common(50)
            if len(term) > 3 and not term.isdigit()
        ]
        
        # Generate sample queries
        sample_queries = [
            "diabetes treatment in children",
            "fever management in infants",
            "asthma diagnosis and therapy",
            "congenital heart disease screening",
            "vaccination schedule recommendations",
            "growth and development milestones",
            "pediatric emergency medicine",
            "infectious disease prevention",
            "nutritional disorders in children",
            "behavioral issues in adolescents"
        ]
        
        print(f"  Generated {len(sample_queries)} sample queries")
        print(f"  Top medical keywords: {', '.join(common_medical_terms[:10])}")
        
        return sample_queries

    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        print("NELSON TEXTBOOK RAG - KNOWLEDGE BASE VALIDATION")
        print("=" * 60)
        
        validation_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'completeness': self.validate_completeness(),
            'token_distribution': self.validate_token_distribution(),
            'content_quality': self.validate_content_quality(),
            'citation_validation': self.validate_citations(),
            'sample_queries': self.generate_sample_queries()
        }
        
        # Overall assessment
        print(f"\n=== OVERALL ASSESSMENT ===")
        total_chunks = validation_results['completeness']['total_chunks']
        avg_confidence = validation_results['content_quality']['confidence_scores']['mean']
        page_coverage = validation_results['completeness']['page_coverage']['coverage_percentage']
        avg_tokens = validation_results['token_distribution']['mean']
        
        print(f"  ✅ Total chunks processed: {total_chunks:,}")
        print(f"  ✅ Page coverage: {page_coverage:.1f}%")
        print(f"  ✅ Average confidence: {avg_confidence:.3f}")
        print(f"  ⚠️  Average tokens per chunk: {avg_tokens:.0f} (target: 500)")
        
        # Quality assessment
        if avg_confidence > 0.8 and page_coverage > 95:
            print(f"  🎉 EXCELLENT: High-quality knowledge base ready for production!")
        elif avg_confidence > 0.7 and page_coverage > 90:
            print(f"  👍 GOOD: Knowledge base suitable for most applications")
        else:
            print(f"  ⚠️  FAIR: Consider improvements before production use")
        
        return validation_results

def main():
    """Main validation function"""
    csv_path = "/project/workspace/nelson_chunks.csv"
    
    validator = KnowledgeBaseValidator(csv_path)
    results = validator.run_full_validation()
    
    # Save validation report
    with open("/project/workspace/validation_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Validation report saved to: validation_report.json")

if __name__ == "__main__":
    main()