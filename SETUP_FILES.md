# Setting Up Nelson Textbook Files

## Required Files

This repository contains the code for processing the Nelson Textbook of Pediatrics, but the actual data files are stored separately due to their large size.

### Files Needed

1. **nelson_chunks.csv** (30.8 MB)
   - Main knowledge base with 4,534 processed chunks
   - Contains all metadata and citations
   - Required for running queries

2. **nelson_textbook_of_pediatrics.pdf** (155.5 MB)
   - Original Nelson Textbook of Pediatrics (22nd Edition)
   - Source material for processing pipeline
   - Optional if you only need to run queries

## Download Instructions

### Method 1: Google Drive (Recommended)

1. Visit the Google Drive folder: [https://drive.google.com/drive/folders/1rJUTvYeOYzesf5WH-BSZpJcq2cEN65mO](https://drive.google.com/drive/folders/1rJUTvYeOYzesf5WH-BSZpJcq2cEN65mO)

2. Download each file individually:
   - Right-click on `nelson_chunks.csv` → Download
   - Right-click on `nelson_textbook_of_pediatrics_2_volum...` → Download

3. Rename the PDF file to: `nelson_textbook_of_pediatrics.pdf`

4. Place both files in this repository directory:
   ```
   DrZ199/Nelson.csv/
   ├── nelson_chunks.csv          ← Place here
   ├── nelson_textbook_of_pediatrics.pdf  ← Place here
   └── ... (other repository files)
   ```

### Method 2: Direct URLs (Advanced)

If you have the specific file IDs, you can use curl:

```bash
# PDF file (known ID from repository)
FILE_ID="1KvjRFW_x-qdXj774UjyO388Ve5lffXgg"
curl -L "https://drive.google.com/uc?export=download&id=$FILE_ID" -o nelson_textbook_of_pediatrics.pdf
```

**Note**: Large files may require additional confirmation steps.

## Verification

After downloading, verify the files:

```bash
# Check CSV file
python3 -c "
import pandas as pd
df = pd.read_csv('nelson_chunks.csv')
print(f'CSV loaded: {len(df)} rows')
print(f'Columns: {list(df.columns)}')
"

# Check PDF file
python3 -c "
import os
if os.path.exists('nelson_textbook_of_pediatrics.pdf'):
    size = os.path.getsize('nelson_textbook_of_pediatrics.pdf') / (1024*1024)
    print(f'PDF file size: {size:.1f} MB')
else:
    print('PDF file not found')
"
```

Expected output:
- CSV: 4,534 rows with proper columns
- PDF: ~155 MB file size

## Usage After Setup

Once files are in place, you can run the query system:

```python
from query_demo import PediatricsQuerySystem

# Initialize with the CSV file
rag = PediatricsQuerySystem("nelson_chunks.csv")

# Test query
results = rag.search("fever in infants", top_k=3)
for result in results:
    print(f"Page {result['page_number']}: {result['summary']}")
```

## Troubleshooting

### File Size Issues
- CSV should be ~30.8 MB
- PDF should be ~155.5 MB
- If files are much smaller, you likely downloaded HTML error pages

### Permission Issues
- Files should be publicly accessible in the Google Drive folder
- No Google account login required

### Path Issues
- Ensure files are in the root of this repository directory
- Use exact filenames: `nelson_chunks.csv` and `nelson_textbook_of_pediatrics.pdf`

## Why Files Are Not In Repository

These files are excluded from the Git repository because:
1. **Size Limits**: GitHub has file size limits (100MB recommended)
2. **Storage Costs**: Large files in Git history increase clone times
3. **Updates**: Data files change independently from code
4. **Distribution**: Google Drive provides better download experience for large files

## File Metadata

### nelson_chunks.csv Schema
```csv
id,book_title,edition,authors,publisher,year,isbn,source_url,drive_link,
page_number,chapter_title,section_title,section_heading_path,chunk_index,
chunk_token_count,chunk_text,chunk_summary,keywords,confidence_score,created_at
```

### Source Information
- **Book**: Nelson Textbook of Pediatrics, 22nd Edition
- **Authors**: Kliegman, Stanton, St. Geme, Schor, Behrman
- **Publisher**: Elsevier (2019)
- **ISBN**: 978-0323529501
- **Pages**: 4,534 pages processed

---

**Need Help?** Create an issue in this repository if you encounter problems downloading or setting up the files.