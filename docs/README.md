# Multi-Language Translation Quality Metrics (MQM) Analyzer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project implements an advanced Multidimensional Quality Metrics (MQM) analyzer for evaluating and improving language translations, supporting multiple language pairs with English as the source language.

## Features

- Upload and analyze English source and multi-language translation files (supports txt, pdf, and docx formats)
- Automatic language detection for the translated text
- Calculate multiple translation quality metrics for various language pairs:
  - BLEU (Adequacy)
  - BERT Score (Precision, Recall, F1-Score for Fluency)
  - METEOR (Content Preservation)
  - Named Entity Recognition Score
  - Semantic Similarity
  - Lexical Similarity
  - BLEURT Score
- Provide initial assessment of translation quality using Azure OpenAI's GPT-4
- Generate revised translations based on the initial assessment
- Compare original and revised translations with detailed insights
- Perform statistical analysis on translation improvements
- Interactive web interface built with Streamlit

## Installation

1. Ensure you have Python 3.10+ installed.

2. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```

3. Clone the repository and navigate to the project directory.

4. Install the project dependencies using Poetry:
   ```
   poetry install
   ```

5. Set up the required NLTK data and spaCy models by running:
   ```
   python setup.py
   ```
   This step is crucial and must be completed before running the application for the first time.

6. Create a `.env` file in the project root and add your Azure OpenAI API credentials:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_API_BASE=your_api_base_url_here
   AZURE_OPENAI_API_VERSION=your_api_version_here
   ```

## Usage

1. Activate the Poetry virtual environment:
   ```
   poetry shell
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

4. Use the web interface to upload your English source file and the translation file in any supported language.

5. Click "Analyze and Improve Translation" to start the analysis process.

6. Review the results, including:
   - Detected language of the translation
   - Original translation scores
   - Initial assessment
   - Revised translation
   - Comparative MQM scores
   - LLM insights
   - Statistical analysis

## Supported Languages

The MQM Analyzer supports translations from English to various languages. The specific language support depends on the available spaCy models and the capabilities of the underlying NLP libraries. Some of the supported languages include:

- Spanish
- French
- German
- Italian
- Portuguese
- Dutch
- Russian
- Chinese
- Japanese
- And many more...

Note: The quality of analysis may vary depending on the language pair and the availability of language-specific resources.

## Dependencies

This project uses several key libraries and tools:

- Streamlit for the web interface
- sacrebleu and bert_score for translation quality metrics
- spaCy for multi-language named entity recognition
- SentenceTransformer for semantic similarity
- NLTK for METEOR score
- Azure OpenAI for GPT-4 powered assessments and improvements
- PyPDF2 and python-docx for handling PDF and DOCX files
- pandas and scikit-learn for statistical analysis
- langdetect for automatic language detection

For a complete list of dependencies, refer to the `pyproject.toml` file.

## Known Limitations and Areas for Future Improvement

1. Language Support: While the tool supports multiple languages, the quality of analysis may vary for less common language pairs or languages with limited NLP resources.

2. Named Entity Recognition: The accuracy of NER may be lower for languages without specific spaCy models. In such cases, the tool falls back to using the English model, which may not be optimal.

3. METEOR Score: The current implementation uses the same METEOR scoring method for all languages, which may not be ideal for languages with significantly different structures from English.

4. BLEURT Score: The BLEURT model is primarily trained on English, so its effectiveness may vary for other languages.

5. Performance: Processing large documents or multiple files simultaneously may be time-consuming. Future improvements could focus on optimization and parallel processing.

6. User Interface: The current Streamlit interface could be enhanced with more interactive features and visualizations of the translation quality metrics.

7. Customization: Future versions could allow users to customize the weights of different metrics or add their own evaluation criteria.

8. Batch Processing: Adding support for analyzing multiple translation pairs in a single run could improve efficiency for large-scale translation projects.

9. API Integration: Developing an API for the tool would allow for easier integration with other translation workflows and systems.

10. Continuous Learning: Implementing a feedback mechanism to improve the tool's assessments based on user input and corrections could enhance its accuracy over time.

## Running Tests

To run the tests for the multi-language support, use the following command:

```
python -m unittest test_multi_language.py
```

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Contributing

Contributions to improve the Multi-Language Translation Quality Metrics (MQM) Analyzer are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements, including support for additional languages or improvements to existing language support.

## Disclaimer

This tool is designed to assist in translation quality assessment and improvement across multiple languages. However, it should not be considered a replacement for professional human translation and review processes. Always verify the results and consult with qualified linguists for critical translations, especially for language pairs where the tool's performance may vary.