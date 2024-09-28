import streamlit as st
import sacrebleu
import bert_score
from typing import Dict
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import nltk
import spacy
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
from bleurt import score
import jellyfish
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import traceback
import time
from functools import wraps
import threading
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Set up logging
logging.basicConfig(level=logging.DEBUG if os.getenv('DEBUG') == 'True' else logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='a')
logger = logging.getLogger(__name__)

# Add a stream handler to also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Custom exception handler for Streamlit
def streamlit_exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    st.error(f"An unexpected error occurred: {str(exc_value)}")

sys.excepthook = streamlit_exception_handler

# Timeout decorator
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function call timed out")]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# Check if NLTK data is available
def check_nltk_data():
    required_data = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words'
    }
    for item, path in required_data.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.warning(f"NLTK data '{item}' not found. Downloading...")
            nltk.download(item)

# Run the check
check_nltk_data()

# Load multilingual SentenceTransformer model
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Initialize BLEURT
bleurt_scorer = score.BleurtScorer()

# Load environment variables
load_dotenv(override=True)

# Configure Azure OpenAI client
try:
    azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_API_BASE is not set in the environment variables")
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=azure_endpoint
    )
    logger.debug(f"Azure OpenAI client configured with endpoint: {azure_endpoint}")
except Exception as e:
    logger.error(f"Error configuring Azure OpenAI client: {str(e)}")
    raise

def read_file_content(file):
    """Read content from uploaded files."""
    logger.info(f"Reading file: {file.name}")
    if file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

def detect_language(text):
    """Detect the language of the given text."""
    return detect(text)

@timeout(30)
def calculate_meteor_score(reference: str, hypothesis: str, lang: str) -> float:
    """Calculate METEOR score for the given reference and hypothesis."""
    logger.info("Calculating METEOR score")
    return nltk.translate.meteor_score.meteor_score([reference.split()], hypothesis.split())

@timeout(30)
def compare_named_entities(source: str, translation: str, source_lang: str, target_lang: str) -> float:
    """Compare named entities between source and translation texts."""
    logger.info("Comparing named entities")
    try:
        source_nlp = spacy.load(f"{source_lang}_core_web_sm")
        target_nlp = spacy.load(f"{target_lang}_core_web_sm")
    except OSError:
        logger.warning(f"SpaCy model for {target_lang} not found. Using English model as fallback.")
        source_nlp = spacy.load("en_core_web_sm")
        target_nlp = source_nlp
    
    source_doc = source_nlp(source)
    translation_doc = target_nlp(translation)
    
    source_entities = set([ent.text for ent in source_doc.ents])
    translation_entities = set([ent.text for ent in translation_doc.ents])
    
    if not source_entities:
        return 1.0  # Perfect score if there are no named entities in the source
    
    common_entities = source_entities.intersection(translation_entities)
    return len(common_entities) / len(source_entities)

@timeout(30)
def calculate_semantic_similarity(source: str, translation: str) -> float:
    """Calculate semantic similarity between source and translation texts."""
    logger.info("Calculating semantic similarity")
    source_embedding = sentence_model.encode(source)
    translation_embedding = sentence_model.encode(translation)
    return pearsonr(source_embedding, translation_embedding)[0]

def calculate_lexical_similarity(source: str, translation: str) -> float:
    """Calculate lexical similarity between source and translation texts."""
    logger.info("Calculating lexical similarity")
    return jellyfish.jaro_winkler_similarity(source, translation)

@timeout(60)
def calculate_mqm_scores(source: str, translation: str, target_lang: str) -> Dict[str, float]:
    """Calculate various MQM scores for the given source and translation texts."""
    logger.info("Calculating MQM scores")
    bleu = sacrebleu.corpus_bleu([translation], [[source]]).score
    P, R, F1 = bert_score.score([translation], [source], lang=target_lang, verbose=False)
    meteor = calculate_meteor_score(source, translation, target_lang)
    ner_score = compare_named_entities(source, translation, "en", target_lang)
    semantic_similarity = calculate_semantic_similarity(source, translation)
    lexical_similarity = calculate_lexical_similarity(source, translation)
    bleurt_score = bleurt_scorer.score(references=[source], candidates=[translation])[0]
    
    return {
        "BLEU (Adequacy)": bleu,
        "BERT Precision": P.mean().item(),
        "BERT Recall": R.mean().item(),
        "BERT F1-Score (Fluency)": F1.mean().item(),
        "METEOR (Content Preservation)": meteor,
        "Named Entity Recognition Score": ner_score,
        "Semantic Similarity": semantic_similarity,
        "Lexical Similarity": lexical_similarity,
        "BLEURT Score": bleurt_score
    }

@timeout(60)
def get_initial_assessment(source: str, original_translation: str, scores: Dict[str, float], target_lang: str) -> str:
    """Get initial assessment of the translation quality using Azure OpenAI."""
    logger.info("Getting initial assessment")
    prompt = f"""
    Analyze the following translation from English to {target_lang}:

    English source: {source}
    {target_lang} translation: {original_translation}

    MQM Scores:
    {', '.join([f'{k}: {v:.2f}' for k, v in scores.items()])}

    Please provide a detailed assessment of the translation quality, including:
    1. Strengths of the translation
    2. Areas for improvement
    3. Specific suggestions for enhancing accuracy and fluency
    4. Analysis of content preservation (METEOR score)
    5. Evaluation of named entity handling
    6. Comments on semantic similarity between source and translation
    7. Lexical similarity analysis
    8. Interpretation of BLEURT score
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI language expert specializing in translation quality assessment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting initial assessment: {str(e)}")
        return f"Error getting initial assessment: {str(e)}"

@timeout(60)
def generate_revised_translation(source: str, original_translation: str, assessment: str, target_lang: str) -> str:
    """Generate a revised translation based on the initial assessment."""
    logger.info("Generating revised translation")
    prompt = f"""
    Based on the following assessment of a {target_lang} translation, please provide an improved version:

    English source: {source}
    Original {target_lang} translation: {original_translation}

    Assessment:
    {assessment}

    Please provide a revised {target_lang} translation that addresses the issues mentioned in the assessment and improves upon the original translation.
    Pay special attention to:
    1. Preserving named entities
    2. Maintaining semantic similarity with the source
    3. Improving content preservation
    4. Enhancing lexical similarity where appropriate
    5. Addressing any issues highlighted by the BLEURT score
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an AI language expert specializing in {target_lang} translation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating revised translation: {str(e)}")
        return f"Error generating revised translation: {str(e)}"

@timeout(60)
def get_azure_openai_insights(source: str, original_translation: str, revised_translation: str, original_scores: Dict[str, float], revised_scores: Dict[str, float], target_lang: str) -> str:
    """Get insights on the original and revised translations using Azure OpenAI."""
    logger.info("Getting Azure OpenAI insights")
    prompt = f"""
    Analyze the following translations from English to {target_lang} and provide insights:

    English source: {source}
    Original {target_lang} translation: {original_translation}
    Revised {target_lang} translation: {revised_translation}

    Original MQM Scores:
    {', '.join([f'{k}: {v:.2f}' for k, v in original_scores.items()])}

    Revised MQM Scores:
    {', '.join([f'{k}: {v:.2f}' for k, v in revised_scores.items()])}

    Please provide insights on:
    1. The quality of both translations
    2. Comparison between the original and revised translations
    3. Any improvements or issues in the revised translation
    4. Interpretation of the MQM scores for both translations, including all metrics
    5. Analysis of how well named entities were preserved in both translations
    6. Evaluation of semantic similarity preservation in both translations
    7. Lexical similarity analysis for both translations
    8. Interpretation of BLEURT scores
    9. Suggestions for further improving the translation process
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI language expert specializing in translation quality assessment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting insights: {str(e)}")
        return f"Error getting insights: {str(e)}"

def perform_statistical_analysis(original_scores: Dict[str, float], revised_scores: Dict[str, float]) -> str:
    """Perform statistical analysis on the original and revised translation scores."""
    logger.info("Performing statistical analysis")
    # Create a DataFrame from the scores
    df = pd.DataFrame({
        'Metric': list(original_scores.keys()),
        'Original': list(original_scores.values()),
        'Revised': list(revised_scores.values())
    })
    
    # Calculate improvements
    df['Improvement'] = df['Revised'] - df['Original']
    
    # Perform K-means clustering on improvements
    scaler = StandardScaler()
    scaled_improvements = scaler.fit_transform(df[['Improvement']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_improvements)
    
    # Generate insights
    insights = "Statistical Analysis Insights:\n\n"
    insights += f"1. Average Improvement: {df['Improvement'].mean():.2f}\n"
    insights += f"2. Metrics with Most Improvement: {', '.join(df.nlargest(3, 'Improvement')['Metric'])}\n"
    insights += f"3. Metrics with Least Improvement: {', '.join(df.nsmallest(3, 'Improvement')['Metric'])}\n"
    insights += "4. Improvement Clusters:\n"
    for cluster in range(3):
        cluster_metrics = df[df['Cluster'] == cluster]['Metric'].tolist()
        insights += f"   Cluster {cluster}: {', '.join(cluster_metrics)}\n"
    
    return insights

def check_language_resources():
    """Check if necessary language resources are installed."""
    try:
        nltk.data.find('tokenizers/punkt')
        spacy.load('en_core_web_sm')
        return True
    except (LookupError, OSError):
        return False

def main():
    st.title("Enhanced Translation Quality Metrics (MQM) Analyzer")

    # Check if language resources are installed
    if not check_language_resources():
        st.error("Required language resources are not installed. Please run 'python setup.py' before using this application.")
        st.stop()

    source_file = st.file_uploader("Upload English source file", type=["txt", "pdf", "docx"])
    translation_file = st.file_uploader("Upload translation file", type=["txt", "pdf", "docx"])

    if st.button("Analyze and Improve Translation"):
        if source_file and translation_file:
            try:
                source_text = read_file_content(source_file)
                original_translation = read_file_content(translation_file)
                
                # Detect the language of the translation
                target_lang = detect_language(original_translation)
                st.info(f"Detected language of translation: {target_lang}")
                
                # Check if the detected language is supported
                supported_languages = ["es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja"]  # Add more as needed
                if target_lang not in supported_languages:
                    st.warning(f"The detected language ({target_lang}) may not be fully supported. Results may be less accurate.")
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(step, total_steps):
                    progress = int((step / total_steps) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Progress: {progress}%")

                total_steps = 7
                current_step = 0

                status_text.text("Calculating original MQM scores...")
                original_scores = calculate_mqm_scores(source_text, original_translation, target_lang)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("Original Translation Scores:")
                st.table([{"Metric": k, "Score": f"{v:.2f}"} for k, v in original_scores.items()])
                
                status_text.text("Getting initial assessment...")
                initial_assessment = get_initial_assessment(source_text, original_translation, original_scores, target_lang)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("Initial Assessment:")
                st.write(initial_assessment)
                
                status_text.text("Generating revised translation...")
                revised_translation = generate_revised_translation(source_text, original_translation, initial_assessment, target_lang)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("Revised Translation:")
                st.write(revised_translation)
                
                status_text.text("Calculating revised MQM scores...")
                revised_scores = calculate_mqm_scores(source_text, revised_translation, target_lang)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("Comparative MQM Scores:")
                comparative_data = []
                for metric in original_scores.keys():
                    original = original_scores[metric]
                    revised = revised_scores[metric]
                    diff = revised - original
                    comparative_data.append({
                        "Metric": metric,
                        "Original": f"{original:.2f}",
                        "Revised": f"{revised:.2f}",
                        "Difference": f"{diff:+.2f}"
                    })
                st.table(comparative_data)

                status_text.text("Getting Azure OpenAI insights...")
                insights = get_azure_openai_insights(source_text, original_translation, revised_translation, original_scores, revised_scores, target_lang)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("LLM Insights:")
                st.write(insights)

                status_text.text("Performing statistical analysis...")
                statistical_analysis = perform_statistical_analysis(original_scores, revised_scores)
                current_step += 1
                update_progress(current_step, total_steps)

                st.subheader("Statistical Analysis:")
                st.write(statistical_analysis)

                status_text.text("Analysis complete!")
                update_progress(total_steps, total_steps)

            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload both source and translation files.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        st.error("An unexpected error occurred. Please check the logs for more information.")
    finally:
        # Perform any necessary cleanup
        logger.info("Cleaning up resources...")
        # Add any specific cleanup operations here if needed