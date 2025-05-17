# Document Information Extractor

This project is a web application built using Streamlit that allows users to extract information from scanned documents, specifically resumes (CVs). The application leverages OCR (Optical Character Recognition) with Tesseract, object detection using YOLO, and NLP techniques for enhanced information extraction.

## Features

* Upload scanned documents (images or PDFs).
* Extract and display information such as name, skills, experience, and more.
* View extracted text in structured JSON format.
* Filter resumes based on specific criteria.
* Train custom YOLO models using uploaded datasets (ZIP or RAR format).
* List all available trained models for use in extraction.
* Display training results (metrics and charts).

## Requirements

* Python 3.8+
* Streamlit
* PIL (Pillow)
* OpenCV
* pytesseract
* ultralytics
* rarfile
* yaml
* textblob
* supervision

## Installation

1. Clone the repository:

```bash
git clone <repo-link>
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py --server.maxUploadSize 1024
```

## Directory Structure

```
.
├── app.py
├── cv_information_extraction.py
├── requirements.txt
└── uploaded_dataset
```

## Usage

* Navigate to the 'Upload & Extract' page to upload scanned documents.
* View extracted data directly on the page.
* Use the 'Model Training' page to upload and train custom YOLO models.
* View trained model metrics and results.

## Configuration

* The application uses a `data.yaml` file to configure dataset paths.
* The training page allows users to set the number of epochs and batch size.

## Troubleshooting

* Ensure the `UnRAR.exe` path is set correctly in `app.py`.
* Make sure all required packages are installed.

