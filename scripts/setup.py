import nltk
import ssl
import subprocess
import sys

def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    print("NLTK data downloaded successfully.")

def download_spacy_models():
    # List of spaCy models to download
    models = [
        "en_core_web_sm",
        "es_core_news_sm",
        "fr_core_news_sm",
        "de_core_news_sm",
        "it_core_news_sm",
        "pt_core_news_sm",
        "nl_core_news_sm",
        "ru_core_news_sm",
        "zh_core_web_sm",
        "ja_core_news_sm"
    ]

    for model in models:
        print(f"Downloading spaCy model: {model}")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])

    print("spaCy models downloaded successfully.")

if __name__ == "__main__":
    download_nltk_data()
    download_spacy_models()

    print("Setup completed successfully. You can now run the Streamlit app using 'streamlit run app.py'.")