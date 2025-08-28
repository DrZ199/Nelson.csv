#!/bin/bash

# Download Google Drive files
# Note: These are large files from the Google Drive folder
# Folder: https://drive.google.com/drive/folders/1rJUTvYeOYzesf5WH-BSZpJcq2cEN65mO

echo "Downloading Nelson Textbook files from Google Drive..."

# Known file ID from repository documentation (PDF file)
PDF_FILE_ID="1KvjRFW_x-qdXj774UjyO388Ve5lffXgg"

# Try to download PDF file
echo "Downloading PDF file..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$PDF_FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$PDF_FILE_ID" -O nelson_textbook_of_pediatrics.pdf && rm -rf /tmp/cookies.txt

echo "Download attempts completed."
echo "If files are missing, please download manually from:"
echo "https://drive.google.com/drive/folders/1rJUTvYeOYzesf5WH-BSZpJcq2cEN65mO"