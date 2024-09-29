# TransLingua IQ

TransLingua IQ is an advanced multi-language translation quality metrics (MQM) analyzer that leverages cutting-edge artificial intelligence and machine learning techniques for accurate and context-aware translation assessment. This tool is designed to revolutionize the way translation quality is evaluated, providing deep insights and actionable improvements for translators, language service providers, and content creators working across multiple languages.

## Project Overview

TransLingua IQ combines natural language processing (NLP) models, machine learning algorithms, and large language models to offer a comprehensive suite of translation quality metrics. By integrating various advanced techniques, the tool provides a nuanced understanding of translation quality that goes beyond traditional metrics, considering factors such as semantic accuracy, fluency, and cultural appropriateness.

## Machine Learning and Large Language Model Techniques

TransLingua IQ utilizes a variety of ML and LLM techniques to achieve its advanced capabilities:

1. **Transformer-based Models**: 
   - BERT (Bidirectional Encoder Representations from Transformers) for contextual word embeddings and semantic understanding.
   - MarianMT for neural machine translation tasks.

2. **Large Language Models**:
   - Integration with Azure OpenAI's GPT-4o for initial assessment and generation of revised translations, leveraging its advanced language understanding and generation capabilities.

3. **Traditional NLP Techniques**:
   - BLEU (Bilingual Evaluation Understudy) for measuring translation adequacy.
   - METEOR (Metric for Evaluation of Translation with Explicit ORdering) for content preservation evaluation.

4. **Advanced Similarity Metrics**:
   - Semantic Similarity using SentenceTransformers for understanding meaning preservation.
   - Lexical Similarity with Jaro-Winkler distance for surface-level text comparison.

5. **Named Entity Recognition (NER)**:
   - Utilizes spaCy's NER models to evaluate the preservation of named entities across translations.

6. **BERT Score**:
   - Leverages BERT embeddings to compute precision, recall, and F1 scores for fluency assessment.

7. **BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)**:
   - A learned metric based on BERT that evaluates translation quality with high correlation to human judgments.

8. **Unsupervised Learning**:
   - K-means clustering for grouping similar improvements in translation quality.

9. **Statistical Analysis**:
   - Utilizes libraries like pandas and scikit-learn for in-depth statistical analysis of translation improvements.

## Features

- **Multi-format Support**: Upload and analyze English source and multi-language translation files (supports txt, pdf, and docx formats).
- **Automatic Language Detection**: Utilizes langdetect for identifying the language of translated text.
- **Comprehensive Quality Metrics**:
  - BLEU (Adequacy)
  - BERT Score (Precision, Recall, F1-Score for Fluency)
  - METEOR (Content Preservation)
  - Named Entity Recognition Score
  - Semantic Similarity
  - Lexical Similarity
  - BLEURT Score
- **AI-Powered Assessment**: Initial assessment of translation quality using Azure OpenAI's GPT-4.
- **Intelligent Revision**: Generation of revised translations based on the initial assessment, leveraging GPT-4's language generation capabilities.
- **Comparative Analysis**: Detailed comparison of original and revised translations with insights.
- **Statistical Insights**: In-depth statistical analysis of translation improvements using machine learning techniques.
- **Interactive Visualization**: User-friendly web interface built with Streamlit for easy interaction and result visualization.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/translingua_iq.git
   cd translingua_iq
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Install required language models:
   ```
   python -m spacy download en_core_web_sm
   python -m spacy download es_core_web_sm
   # Add commands for other language models as needed
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_API_BASE=your_azure_openai_api_base
   AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
   ```

## Usage

1. Ensure all dependencies and language models are installed.
2. Run the Streamlit app:
   ```
   poetry run streamlit run translingua_iq/src/app.py
   ```
3. Navigate to the provided local URL in your web browser.
4. Upload your source and translation files.
5. Interact with the TransLingua IQ interface to analyze and improve translations.

## Contributing

We welcome contributions to TransLingua IQ! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Changelog

For a detailed history of changes and version updates, please refer to the [CHANGELOG.md](CHANGELOG.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes various open-source libraries and pre-trained models. We're grateful to the AI and NLP research community for their invaluable contributions.
- Special thanks to the Azure OpenAI team for providing access to the GPT-4 model, which significantly enhances our translation assessment capabilities.