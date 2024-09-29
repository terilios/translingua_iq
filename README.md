# TransLingua IQ

TransLingua IQ is an advanced multi-language translation quality metrics (MQM) analyzer leveraging artificial intelligence for accurate and context-aware translation assessment.

## Features

- Upload and analyze English source and multi-language translation files (supports txt, pdf, and docx formats)
- Automatic language detection for translated text
- Calculation of multiple translation quality metrics:
  - BLEU (Adequacy)
  - BERT Score (Precision, Recall, F1-Score for Fluency)
  - METEOR (Content Preservation)
  - Named Entity Recognition Score
  - Semantic Similarity
  - Lexical Similarity
  - BLEURT Score
- Initial assessment of translation quality using Azure OpenAI's GPT-4
- Generation of revised translations based on the initial assessment
- Comparison of original and revised translations with detailed insights
- Statistical analysis on translation improvements
- Interactive web interface built with Streamlit

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/translingua_iq.git
   cd translingua_iq
   ```

2. Install dependencies:
   ```
   poetry install
   ```

3. Install the Spanish language model for spaCy:
   ```
   python -m spacy download es_core_news_sm
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_API_BASE=your_azure_openai_api_base
   AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
   ```

## Usage

Run the Streamlit app:
```
poetry run streamlit run translingua_iq/src/app.py
```

Navigate to the provided local URL in your web browser to use the TransLingua IQ interface.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Changelog

See the [CHANGELOG.md](CHANGELOG.md) file for details on recent changes and version history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.