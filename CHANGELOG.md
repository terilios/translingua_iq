# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial release of the Multi-Language Translation Quality Metrics (MQM) Analyzer.
- Upload and analyze English source and multi-language translation files (supports txt, pdf, and docx formats).
- Automatic language detection for the translated text.
- Calculation of multiple translation quality metrics for various language pairs:
  - BLEU (Adequacy)
  - BERT Score (Precision, Recall, F1-Score for Fluency)
  - METEOR (Content Preservation)
  - Named Entity Recognition Score
  - Semantic Similarity
  - Lexical Similarity
  - BLEURT Score
- Initial assessment of translation quality using Azure OpenAI's GPT-4.
- Generation of revised translations based on the initial assessment.
- Comparison of original and revised translations with detailed insights.
- Statistical analysis on translation improvements.
- Interactive web interface built with Streamlit.

### Changed
- Updated project dependencies in pyproject.toml.
- Improved error handling for Azure OpenAI API calls.

### Fixed
- Resolved issues with Spanish language model in spaCy by installing es_core_news_sm.

## [0.1.0] - 2023-09-28
### Added
- Initial release of the Multi-Language Translation Quality Metrics (MQM) Analyzer.
