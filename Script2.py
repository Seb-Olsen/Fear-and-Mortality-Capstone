#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical Analysis Script - ECCO TEI (London/Genre Filtered)
Analyzes correlation between YEARLY AVERAGE/MAX/PROPORTION fear sentiment
(derived using MacBERTh embeddings) and MONTHLY mortality trends using TFT forecasting.

*** TARGET ENVIRONMENT: Python 3.10+, pytorch-forecasting (latest compatible),
                       pytorch-lightning (latest compatible), transformers, sentence-transformers,
                       torch, nltk, pandas, numpy, matplotlib, seaborn, joblib, symspellpy, statsmodels ***

*** NOTE: Uses yearly sentiment mapped to monthly data due to ECCO date limitations. ***
"""

import os
import logging
import re
import warnings
from datetime import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Union, List, Optional

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings("ignore", ".*does not have valid feature names*") # Sklearn scaler warning

# ---------------------
# Basic Setup
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------
# NLTK Setup & Download
# ---------------------
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    logger.info("Checking/Downloading required NLTK data...")
    required_packages = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('stopwords', 'corpora/stopwords'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    downloader = nltk.downloader.Downloader()
    for package_id, path in required_packages:
        package_found = False
        try:
            nltk.data.find(path) # Check without quiet=True
            logger.debug(f"NLTK data '{package_id}' found.")
            package_found = True
        except LookupError:
             package_found = False
        except Exception as e:
             logger.warning(f"NLTK check failed for '{package_id}': {e}")
             package_found = False

        if not package_found:
            logger.info(f"Downloading NLTK package: {package_id}")
            try:
                force_dl = (package_id == 'punkt')
                if not downloader.download(package_id, quiet=True, force=force_dl):
                    try:
                        nltk.data.find(path)
                        logger.info(f"NLTK '{package_id}' download completed or found after check.")
                    except LookupError:
                        raise RuntimeError(f"Failed to download or locate NLTK package: {package_id} at path {path}")
                else:
                    logger.info(f"NLTK '{package_id}' downloaded successfully.")
            except Exception as e:
                logger.error(f"NLTK download error for {package_id}: {e}", exc_info=True)
                raise

download_nltk_data()

# ---------------------
# Spelling / Transformers / PyTorch / Embeddings
# ---------------------
from symspellpy.symspellpy import SymSpell, Verbosity
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---------------------
# Time-series, ML
# ---------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import grangercausalitytests

# ---------------------
# PyTorch Forecasting for TFT
# ---------------------
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, EncoderNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

# ---------------------
# Caching Setup
# ---------------------
from joblib import Memory
cache_dir = "cache_dir_monthly_macberth_filtered" # New cache for potentially filtered data
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)

# ---------------------
# Configuration
# ---------------------
ECCO_DIR = '/Users/sebo/Desktop/AUC/Semester 6/Capstone/Programming/ecco_all/p5/ecco_p5_released' # Adjust if needed
COUNTS_FILE = 'WeeklyBillsMortality1644to1849/counts.txt' # Adjust if needed
HISTORICAL_DICT_PATH = "historical_dict.txt" # for SymSpell
START_YEAR = 1701
END_YEAR = 1800
INFECTIOUS_DISEASE_TYPES = None # Use None for total mortality (excluding christened)

# --- Genre Filtering ---
RELEVANT_GENRES = [
    "medicine", "treatise", "treatises", "medical", # Medical terms
    "newsbooks", "newspapers", "periodicals", "news", # News/Current Events
    "advertisements", "advertisement",
    "pamphlets", "pamphlet",
    "letters", "letter", "correspondence", # Broader contemporary accounts
    "sermons", "sermon",
    "miscellany", "miscellanies"
]
RELEVANT_GENRES = list(set([g.lower() for g in RELEVANT_GENRES])) # Lowercase and unique

# --- Sentiment Analysis Configuration ---
MACBERTH_MODEL_NAME = 'emanjavacas/MacBERTh'
FEAR_WORDS = [
    "fear", "afraid", "scared", "terror", "dread", "panic", "anxiety", "worry",
    "horror", "phobia", "fright", "alarm", "apprehension", "nervousness",
    "trembling", "timidity", "consternation", "distress", "unease",
    "pestilence", "plague", "contagion", "infection", "epidemic",
    "sickness", "disease", "illness", "malady", "distemper",
    "dying", "death", "mortality", "fatal", "corpse", "grave", "burial",
    "suffering", "agony", "misery", "calamity", "crisis", "danger"
]
FEAR_WORDS = list(set(FEAR_WORDS)) # Unique words

# --- Aggregation Method ---
# Choose *one* method for yearly sentiment aggregation
# Options: 'mean', 'max', 'proportion'
AGGREGATION_METHOD = 'mean'
FEAR_THRESHOLD = 0.75 # Used only if AGGREGATION_METHOD is 'proportion'

# --- TFT Parameters (Monthly) ---
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.2
HIDDEN_CONTINUOUS_SIZE = 16
MONTHLY_MAX_ENCODER_LENGTH = 24
MONTHLY_MAX_PREDICTION_LENGTH = 6
GRADIENT_CLIP_VAL = 0.15

# --- Plotting Directory ---
PLOT_DIR = "plots_monthly_macberth_filtered" # New plot dir
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

# --- Device Setup ---
logger.info("Forcing CPU due to potential MPS compatibility issues or preference.")
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# --- XML Namespace ---
# Double-check the root element of XML files if parsing fails later
XML_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

# -----------------------------------------------------------------------------
# 1. Text Normalization & Preprocessing Helpers 
# -----------------------------------------------------------------------------
def normalize_historical_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("ſ", "s")
    return text

def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'): return wn.ADJ
    elif treebank_tag.startswith('V'): return wn.VERB
    elif treebank_tag.startswith('N'): return wn.NOUN
    elif treebank_tag.startswith('R'): return wn.ADV
    else: return wn.NOUN

# -----------------------------------------------------------------------------
# 2. SymSpell Setup & Correction (WIP)
# -----------------------------------------------------------------------------
@memory.cache
def setup_symspell(dictionary_path=HISTORICAL_DICT_PATH, max_edit_distance=1):
    if not os.path.exists(dictionary_path):
        logger.warning(f"SymSpell dict not found: {dictionary_path}. Skipping correction.")
        return None
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        try: loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
        except UnicodeDecodeError: logger.warning("UTF-8 failed for dict, try latin-1."); loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='latin-1')
        if not loaded: logger.error(f"Failed to load SymSpell dict: {dictionary_path}"); return None
        logger.info(f"SymSpell dictionary loaded (max_edit_distance={max_edit_distance}).")
        return sym_spell
    except Exception as e: logger.error(f"Error setting up SymSpell: {e}", exc_info=True); return None

sym_spell_global = setup_symspell()

def correct_ocr_spelling(text: str, sym_spell: Optional[SymSpell]) -> str:
    if not sym_spell or not isinstance(text, str) or not text.strip(): return text
    words = text.split(); corrected_words = []
    for word in words:
        clean_word = word.strip('.,!?;:"()[]')
        if not clean_word or not clean_word.isalpha(): corrected_words.append(word); continue
        suggestions = sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=sym_spell.max_dictionary_edit_distance, include_unknown=True)
        if suggestions:
            best_suggestion = suggestions[0].term
            apply_correction = best_suggestion.lower() != clean_word.lower()
            if apply_correction:
                if word.istitle() and len(word) > 1: corrected_words.append(best_suggestion.capitalize())
                elif word.isupper() and len(word) > 1: corrected_words.append(best_suggestion.upper())
                else: corrected_words.append(best_suggestion)
            else: corrected_words.append(word)
        else: corrected_words.append(word)
    return " ".join(corrected_words)

# -----------------------------------------------------------------------------
# 3. Core Text Preprocessing Function  
# -----------------------------------------------------------------------------
def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set, sym_spell: Optional[SymSpell] = None, use_symspell: bool = False) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = normalize_historical_text(text)
    if use_symspell and sym_spell: text = correct_ocr_spelling(text, sym_spell)
    text_cleaned = re.sub(r"[^\w\s]", " ", text); text_cleaned = re.sub(r"\d+", "", text_cleaned)
    text_cleaned = text_cleaned.lower(); text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    if not text_cleaned: return ""
    try: tokens = nltk.word_tokenize(text_cleaned)
    except Exception as e: logger.warning(f"NLTK Tokenization failed: {e}. Text: '{text_cleaned[:50]}...'"); return ""
    if not tokens: return ""
    try: tagged_tokens = nltk.pos_tag(tokens)
    except Exception as e: logger.warning(f"NLTK POS Tagging failed: {e}. Tokens: {tokens[:5]}. Defaulting noun."); tagged_tokens = [(t, 'NN') for t in tokens]
    processed_tokens = []
    for word, tag in tagged_tokens:
        if word not in stop_words and len(word) > 1:
            try: lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)); processed_tokens.append(lemma)
            except Exception as e: logger.warning(f"Lemmatization failed for '{word}'/'{tag}': {e}"); processed_tokens.append(word)
    return " ".join(processed_tokens)

# -----------------------------------------------------------------------------
# 4. Cached Data Loading and Processing Functions
# -----------------------------------------------------------------------------

# === ECCO Parsing ===
@memory.cache
def parse_ecco_tei_files(ecco_dir: str = ECCO_DIR, start_year: int = START_YEAR, end_year: int = END_YEAR, relevant_genres: list = RELEVANT_GENRES) -> pd.DataFrame:
    """
    (REVISED - NEEDS XML INSPECTION) Parses ECCO TEI files, extracts YEAR, text,
    filters for London place AND relevant genres.
    !!! REQUIRES ADJUSTING XPATH EXPRESSIONS based on manual XML inspection !!!
    Returns DataFrame with ['year', 'text', 'genre', 'doc_id'].
    """
    records = []
    logger.info(f"Starting ECCO TEI parsing (Years: {start_year}-{end_year}, Place: London, Genres: {'/'.join(relevant_genres[:5])}...).")
    file_count = 0; processed_count = 0; skipped_genre = 0; skipped_place = 0
    skipped_year = 0; skipped_text = 0; parse_errors = 0

    ns = XML_NS # Use configured namespace
    relevant_genres_lower = set(g.lower() for g in relevant_genres)

    PLACE_XPATHS_TO_TRY = [
        './/tei:sourceDesc/tei:biblFull/tei:publicationStmt/tei:pubPlace'
    ]


    for rootdir, _, files in os.walk(ecco_dir):
        for fname in files:
            if not fname.endswith('.xml'): continue
            file_count += 1
            if file_count % 5000 == 0: logger.info(f" Scanning file {file_count}...")

            fpath = os.path.join(rootdir, fname)
            doc_id = os.path.splitext(fname)[0]

            try:
                tree = ET.parse(fpath)
                root = tree.getroot()

                # --- Extract Year ---
                year_val = None; date_text_found = None
                date_paths = ['.//tei:publicationStmt/tei:date', './/tei:sourceDesc//tei:date']
                for path in date_paths:
                    for date_elem in root.findall(path, ns): # Use findall to catch multiple date tags if they exist
                         if date_elem is not None and date_elem.text:
                             date_text_candidate = date_elem.text.strip()
                             year_match = re.search(r'\b(1[6-8]\d{2})\b', date_text_candidate)
                             if year_match:
                                 year_val_candidate = int(year_match.group(1))
                                 if start_year <= year_val_candidate <= end_year:
                                     year_val = year_val_candidate; date_text_found = date_text_candidate
                                     break # Found a valid year in this path
                    if year_val: break # Stop checking paths if valid year found
                if year_val is None: # Fallback search in header
                    header = root.find('.//tei:teiHeader', ns)
                    if header is not None:
                        header_text = ''.join(t for t in header.itertext() if t)
                        year_matches = re.findall(r'\b(1[6-8]\d{2})\b', header_text)
                        valid_years = [int(yr) for yr in year_matches if start_year <= int(yr) <= end_year]
                        if valid_years: year_val = min(valid_years) # Take earliest

                if year_val is None: skipped_year += 1; continue

                # --- Filter by Publication Place (London) ---
                is_london = False; found_place_text = None
                if not PLACE_XPATHS_TO_TRY: logger.warning("PLACE_XPATHS_TO_TRY is empty. Place filtering skipped."); skipped_place +=1; continue
                for xpath in PLACE_XPATHS_TO_TRY:
                    place_nodes = root.findall(xpath, ns)
                    for place_node in place_nodes:
                         node_text = (''.join(t for t in place_node.itertext() if t) or "").strip()
                         if node_text:
                              found_place_text = node_text # Log first found place
                              # More robust London check, avoiding digitizer info
                              if "london" in node_text.lower() and "ann arbor" not in node_text.lower() and "michigan" not in node_text.lower():
                                  is_london = True; break
                    if is_london: break
                if not is_london: skipped_place += 1; continue

                # # --- Filter by Genre ---
                doc_genre = "not_filtered"; genre_found = True; 
                # if not GENRE_XPATHS_TO_TRY: logger.warning("GENRE_XPATHS_TO_TRY is empty. Genre filtering skipped."); skipped_genre += 1; continue
                # for xpath in GENRE_XPATHS_TO_TRY:
                #     genre_nodes = root.findall(xpath, ns)
                #     for genre_node in genre_nodes:
                #          term = None
                #          if genre_node.text: term = genre_node.text.strip().lower()
                #          elif 'target' in genre_node.attrib: term = genre_node.attrib['target'].split('/')[-1].lower() # Example for catRef
                #          if term:
                #              actual_genres_in_doc.append(term)
                #              if term in relevant_genres_lower:
                #                  genre_found = True; doc_genre = term; break
                #     if genre_found: break
                # if not genre_found: skipped_genre += 1; continue

                # --- Extract Text ---
                text_nodes = root.findall('.//tei:text', ns); full_text_parts = []
                for node in text_nodes:
                    node_text = ' '.join(t.strip() for t in node.itertext() if t and t.strip())
                    node_text_clean = re.sub(r'\s+', ' ', node_text).strip()
                    if node_text_clean: full_text_parts.append(node_text_clean)
                if not full_text_parts: skipped_text += 1; continue
                combined_text = ' '.join(full_text_parts)

                # Append record if ALL filters passed
                records.append({'year': year_val, 'text': combined_text, 'doc_id': doc_id})
                processed_count += 1
                if processed_count % 500 == 0: logger.info(f" Found {processed_count} relevant documents (parsed {file_count} files)...")

            except ET.ParseError as e: logger.warning(f"XML Parse Error in {fname}: {e}"); parse_errors += 1
            except Exception as e: logger.warning(f"General Error processing {fname}: {e}", exc_info=False); parse_errors += 1

    logger.info(f"Finished ECCO parsing. Total files scanned: {file_count}")
    if processed_count == 0:
         logger.error("CRITICAL: No ECCO documents found matching ALL criteria (Year, Place, Genre). Check XPaths, genre list, place filter, year range, or XML structure.")
    else:
         logger.info(f" Found {processed_count} valid 'London' records matching genres.")
    logger.info(f" Skipped: {skipped_year} (year), {skipped_place} (place), {skipped_genre} (genre), {skipped_text} (no text).")
    logger.info(f" Errors during parsing: {parse_errors}")

    if not records: return pd.DataFrame(columns=["year", "text", "genre", "doc_id"])

    df = pd.DataFrame(records)
    df['text'] = df['text'].astype(str)
    logger.info(f"ECCO DataFrame prepared: {df.shape[0]} records. Year Range: {df['year'].min()} to {df['year'].max()}")
    return df


# === Mortality Loading (MONTHLY - Unchanged) ===
def parse_weekID_to_monthly(week_str):
    try:
        year_str, week_ = week_str.split("/")
        year = int(year_str); week = int(week_)
        if not (START_YEAR <= year <= END_YEAR) or not (1 <= week <= 53): return None
        try: week_start_date = datetime.fromisocalendar(year, week, 1)
        except ValueError:
             try: week_start_date = datetime.fromisocalendar(year, 52, 1)
             except ValueError: logger.debug(f"Invalid week {week_} for year {year_str}. Skip."); return None
        return week_start_date.replace(day=1)
    except Exception as e: logger.debug(f"Error parsing weekID '{week_str}' to monthly: {e}"); return None

@memory.cache
def load_and_aggregate_monthly_mortality(file_path: str = COUNTS_FILE, disease_types: Optional[List[str]] = INFECTIOUS_DISEASE_TYPES, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    if not os.path.exists(file_path): raise FileNotFoundError(f"Mortality file not found: {file_path}")
    logger.info(f"Loading mortality data from {file_path}...")
    df = pd.read_csv(file_path, delimiter="|", low_memory=False, dtype={'weekID': str})
    required_cols = ["weekID", "counttype", "countn"]
    if not all(c in df.columns for c in required_cols): raise ValueError(f"Mortality file missing required columns")
    df["countn"] = pd.to_numeric(df["countn"], errors="coerce")
    original_len = len(df); df.dropna(subset=["countn"], inplace=True); df["countn"] = df["countn"].astype(int)
    if len(df) < original_len: logger.warning(f"Dropped {original_len - len(df)} rows non-numeric 'countn'.")
    logger.info("Parsing week IDs to month start dates...")
    df["month_date"] = df["weekID"].astype(str).apply(parse_weekID_to_monthly)
    original_len = len(df); df.dropna(subset=["month_date"], inplace=True)
    if (original_len - len(df)) > 0: logger.info(f"Dropped {original_len - len(df)} rows invalid weekID/date.")
    df["year"] = df["month_date"].dt.year
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    if df.empty: logger.warning(f"No mortality records in range {start_year}-{end_year}."); return pd.DataFrame(columns=["month_date", "year", "month", "deaths"])
    logger.info(f"{len(df)} weekly records after date range filter.")
    df['counttype'] = df['counttype'].str.lower().str.strip()
    df = df[df["counttype"] != "christened"]
    if disease_types: df = df[df["counttype"].isin([d.lower() for d in disease_types])]
    else: logger.info("Using total mortality (excluding 'christened').")
    if df.empty: logger.warning("No records after type filter."); return pd.DataFrame(columns=["month_date", "year", "month", "deaths"])
    logger.info("Aggregating weekly counts to MONTHLY mortality totals...")
    monthly_sum = df.groupby("month_date")["countn"].sum().reset_index()
    monthly_sum.rename(columns={"countn": "deaths"}, inplace=True)
    monthly_sum["deaths"] = monthly_sum["deaths"].astype(float)
    monthly_sum["year"] = monthly_sum["month_date"].dt.year
    monthly_sum["month"] = monthly_sum["month_date"].dt.month
    monthly_sum = monthly_sum.sort_values("month_date").reset_index(drop=True)
    logger.info(f"Mortality aggregated: {monthly_sum.shape[0]} months. Range: {monthly_sum['month_date'].min():%Y-%m} to {monthly_sum['month_date'].max():%Y-%m}")
    return monthly_sum[["month_date", "year", "month", "deaths"]]

# === Text Preprocessing   ===
@memory.cache
def preprocess_text_dataframe(df: pd.DataFrame, text_col: str = "text", use_symspell: bool = False) -> pd.DataFrame:
    logger.info(f"Preprocessing text column '{text_col}' (use_symspell={use_symspell})...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")
    df_copy = df.copy(); df_copy[text_col] = df_copy[text_col].astype(str).fillna('')
    lemmatizer = WordNetLemmatizer(); stop_words = set(stopwords.words("english"))
    global sym_spell_global
    total_rows = len(df_copy); processed_texts = []
    for i, text in enumerate(df_copy[text_col]):
         if (i + 1) % 5000 == 0: logger.info(f" Preprocessing text {i+1}/{total_rows}...")
         processed = preprocess_text(text, lemmatizer, stop_words, sym_spell=sym_spell_global, use_symspell=use_symspell)
         processed_texts.append(processed)
    df_copy['processed_text'] = processed_texts
    original_len = len(df_copy); df_copy = df_copy[df_copy['processed_text'].str.strip().astype(bool)]
    if len(df_copy) < original_len: logger.warning(f"Dropped {original_len - len(df_copy)} rows due to empty processed text.")
    logger.info(f"Text preprocessing complete. Shape: {df_copy.shape}")
    return df_copy

# === Fear Scoring using MacBERTh (Unchanged Class Definition) ===
class MacBERThFearScorer:
    _instance = None; _model_name = MACBERTH_MODEL_NAME; _fear_words = FEAR_WORDS
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: logger.info(f"Creating MacBERThFearScorer instance..."); cls._instance = super().__new__(cls); cls._instance._initialized = False
        return cls._instance
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if self._initialized: return
        logger.info(f"Initializing MacBERTh model for embedding: {self._model_name}...")
        self.device = device if device else DEVICE # Use global DEVICE default
        logger.info(f"MacBERTh will run on device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name).to(self.device); self.model.eval()
            logger.info("Calculating average fear vector embedding...")
            with torch.no_grad():
                fear_inputs = self.tokenizer(self._fear_words, padding=True, truncation=True, return_tensors="pt").to(self.device)
                fear_outputs = self.model(**fear_inputs)
                fear_embeddings = self._mean_pooling(fear_outputs, fear_inputs['attention_mask']).cpu().numpy()
            self.average_fear_vector = np.mean(fear_embeddings, axis=0).reshape(1, -1)
            logger.info(f"Average fear vector calculated (shape: {self.average_fear_vector.shape}).")
            self._initialized = True
        except Exception as e: logger.error(f"Failed to initialize MacBERTh: {e}", exc_info=True); raise
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]; input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1); sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    @torch.no_grad()
    def calculate_fear_scores(self, texts: List[str], batch_size: int = 32) -> List[float]:
        if not self._initialized: raise RuntimeError("MacBERThFearScorer not initialized.")
        if not texts: return []
        all_fear_scores = []; num_texts = len(texts); logger.info(f"Calculating fear scores for {num_texts} texts using MacBERTh...")
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i : i + batch_size]; valid_batch_texts = [str(t) if t else "" for t in batch_texts]
            inputs = self.tokenizer(valid_batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
            similarities = cosine_similarity(embeddings, self.average_fear_vector)
            all_fear_scores.extend(similarities.flatten().tolist())
            if (i // batch_size + 1) % (max(1, (num_texts // batch_size) // 10)) == 0: logger.info(f" Scored {min(i + batch_size, num_texts)}/{num_texts}...")
        if len(all_fear_scores) != num_texts: logger.error(f"Fear score length mismatch: {len(all_fear_scores)} vs {num_texts}. Padding."); all_fear_scores.extend([0.0] * (num_texts - len(all_fear_scores)))
        logger.info("Fear score calculation complete.")
        return all_fear_scores

@memory.cache
def calculate_fear_scores_dataframe(df: pd.DataFrame, text_col: str = "processed_text", batch_size: int = 32) -> pd.DataFrame:
    logger.info(f"Calculating MacBERTh-based fear scores for column '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found for fear scoring.")
    if df[text_col].isnull().any(): logger.warning(f"'{text_col}' contains NaNs. Filling with empty string."); df[text_col] = df[text_col].fillna('')
    df_copy = df.copy(); scorer = MacBERThFearScorer(); texts_to_score = df_copy[text_col].tolist()
    fear_scores = scorer.calculate_fear_scores(texts_to_score, batch_size=batch_size)
    df_copy["fear_score"] = fear_scores
    logger.info("Fear scoring complete. Added 'fear_score' column.")
    logger.info(f"Fear score stats: Min={np.min(fear_scores):.3f}, Max={np.max(fear_scores):.3f}, Mean={np.mean(fear_scores):.3f}, Std={np.std(fear_scores):.3f}")
    return df_copy

# === Data Aggregation and Merging (Handles Different Aggregations) ===
@memory.cache
def aggregate_yearly_sentiment_and_merge_monthly(ecco_df: pd.DataFrame, monthly_mortality_df: pd.DataFrame, agg_method: str = AGGREGATION_METHOD, fear_thresh: float = FEAR_THRESHOLD) -> pd.DataFrame:
    """
    (Cached) Aggregates YEARLY fear scores using chosen method ('mean', 'max', 'proportion'),
    then merges this yearly score onto the MONTHLY mortality data. Creates lagged features.
    Standardizes the chosen feature and its lag.
    Returns a monthly DataFrame ready for TFT.
    """
    logger.info(f"Aggregating YEARLY sentiment using '{agg_method}' and merging with MONTHLY mortality...")
    if 'fear_score' not in ecco_df.columns or 'year' not in ecco_df.columns: raise ValueError("'fear_score','year' needed in ecco_df.")
    if not all(c in monthly_mortality_df.columns for c in ['month_date', 'year', 'month', 'deaths']): raise ValueError("mortality_df missing cols.")

    ecco_df_copy = ecco_df[['year', 'fear_score']].copy()
    ecco_df_copy['fear_score'] = pd.to_numeric(ecco_df_copy['fear_score'], errors='coerce')
    ecco_df_copy.dropna(subset=['fear_score'], inplace=True)

    # --- Aggregate Fear Score by YEAR using chosen method ---
    yearly_sentiment = pd.DataFrame()
    feature_col_name = f'fear_score_yearly_{agg_method}' # e.g., fear_score_yearly_mean

    if agg_method == 'mean':
        yearly_sentiment = ecco_df_copy.groupby('year')['fear_score'].mean().reset_index()
    elif agg_method == 'max':
        yearly_sentiment = ecco_df_copy.groupby('year')['fear_score'].max().reset_index()
    elif agg_method == 'proportion':
        logger.info(f"Calculating proportion of docs with fear_score > {fear_thresh}")
        ecco_df_copy['high_fear'] = (ecco_df_copy['fear_score'] > fear_thresh).astype(int)
        yearly_sentiment = ecco_df_copy.groupby('year')['high_fear'].mean().reset_index() # Mean of 0/1 gives proportion
    else:
        raise ValueError(f"Unsupported aggregation method: {agg_method}. Choose 'mean', 'max', or 'proportion'.")

    yearly_sentiment.rename(columns={yearly_sentiment.columns[1]: feature_col_name}, inplace=True)
    logger.info(f"Aggregated yearly sentiment ({agg_method}) for {yearly_sentiment.shape[0]} years.")

    # --- Merge Yearly Aggregated Fear Score onto Monthly Mortality Data ---
    logger.info(f"Merging yearly '{feature_col_name}' onto monthly mortality data...")
    yearly_sentiment['year'] = yearly_sentiment['year'].astype(int)
    monthly_mortality_df['year'] = monthly_mortality_df['year'].astype(int)
    merged_df = pd.merge(monthly_mortality_df, yearly_sentiment, on='year', how='left')

    # Handle potential missing years in sentiment
    missing_rows = merged_df[feature_col_name].isnull().sum()
    if missing_rows > 0:
        logger.warning(f"{missing_rows} monthly rows have missing yearly sentiment. Imputing with global mean/median.")
        fill_value = yearly_sentiment[feature_col_name].median() # Use median for robustness
        merged_df[feature_col_name].fillna(fill_value, inplace=True)

    # --- Create Lagged Yearly Fear Score ---
    feature_lag1_col_name = f'{feature_col_name}_lag1' # e.g., fear_score_yearly_mean_lag1
    logger.info(f"Creating lagged feature: '{feature_lag1_col_name}'...")
    merged_df = merged_df.sort_values("month_date").reset_index(drop=True)
    merged_df[feature_lag1_col_name] = merged_df[feature_col_name].shift(12)
    initial_nan_count = merged_df[feature_lag1_col_name].isnull().sum()
    if initial_nan_count > 0:
        logger.info(f"Imputing {initial_nan_count} initial NaNs in lagged fear score.")
        merged_df[feature_lag1_col_name].fillna(method='bfill', inplace=True)
        merged_df[feature_lag1_col_name].fillna(merged_df[feature_lag1_col_name].median(), inplace=True) # Fill remaining with median

    # --- Standardize the Chosen Fear Scores ---
    # Use generic names for standardized columns for easier use in TFT config
    feature_std_col_name = 'feature_std'
    feature_lag1_std_col_name = 'feature_lag1_std'
    logger.info(f"Standardizing '{feature_col_name}' -> '{feature_std_col_name}' and '{feature_lag1_col_name}' -> '{feature_lag1_std_col_name}'.")

    scaler_current = StandardScaler()
    merged_df[feature_std_col_name] = scaler_current.fit_transform(merged_df[[feature_col_name]])

    scaler_lagged = StandardScaler()
    # Ensure no NaNs before scaling lag (should be handled above)
    merged_df[feature_lag1_std_col_name] = scaler_lagged.fit_transform(merged_df[[feature_lag1_col_name]])

    # --- Prepare for TFT (Monthly Time Index) ---
    merged_df = merged_df.sort_values("month_date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["month_date"] - merged_df["month_date"].min()).dt.days // 30
    merged_df["deaths"] = merged_df["deaths"].astype(float)
    merged_df[feature_std_col_name] = merged_df[feature_std_col_name].astype(float)
    merged_df[feature_lag1_std_col_name] = merged_df[feature_lag1_std_col_name].astype(float)
    merged_df["year"] = merged_df["year"].astype(str)
    merged_df["month"] = merged_df["month"].astype(str)
    merged_df["series_id"] = "London"

    # Select and order columns for clarity
    final_cols = [
        "month_date", "time_idx", "deaths", "year", "month", "series_id",
        feature_col_name, feature_lag1_col_name, # Raw chosen scores
        feature_std_col_name, feature_lag1_std_col_name # Standardized scores for model
    ]
    # Ensure all columns exist before selecting
    final_cols = [col for col in final_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols]

    logger.info(f"Final monthly data shape for TFT: {merged_df.shape}. Time idx range: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Columns: {merged_df.columns.tolist()}")
    logger.info(f"NaN check:\n{merged_df.isnull().sum()}")
    if merged_df["time_idx"].max() < MONTHLY_MAX_ENCODER_LENGTH + MONTHLY_MAX_PREDICTION_LENGTH:
         raise ValueError(f"Insufficient monthly data span ({merged_df.shape[0]} months) for TFT config.")
    return merged_df


# -----------------------------------------------------------------------------
# 5. TFT Training and Evaluation (Monthly - Using Default Norm)
# -----------------------------------------------------------------------------
def train_tft_model(df: pd.DataFrame,
                    max_epochs: int = MAX_EPOCHS,
                    batch_size: int = BATCH_SIZE,
                    encoder_length: int = MONTHLY_MAX_ENCODER_LENGTH,
                    pred_length: int = MONTHLY_MAX_PREDICTION_LENGTH,
                    lr: float = LEARNING_RATE,
                    hidden_size: int = HIDDEN_SIZE,
                    attn_heads: int = ATTENTION_HEAD_SIZE,
                    dropout: float = DROPOUT,
                    hidden_cont_size: int = HIDDEN_CONTINUOUS_SIZE,
                    clip_val: float = GRADIENT_CLIP_VAL) -> Tuple[Optional[TemporalFusionTransformer], Optional[pl.Trainer], Optional[torch.utils.data.DataLoader], Optional[TimeSeriesDataSet]]:
    """Trains the Temporal Fusion Transformer model on MONTHLY data using default scaling."""

    logger.info(f"Setting up MONTHLY TFT model training...")
    logger.info(f" Encoder length: {encoder_length} months, Prediction length: {pred_length} months")

    max_idx = df["time_idx"].max(); training_cutoff = max_idx - pred_length
    logger.info(f"Monthly Data Cutoff for Training: time_idx <= {training_cutoff}")
    if training_cutoff < encoder_length: logger.error(f"Cutoff {training_cutoff} < Encoder length {encoder_length}."); return None, None, None, None

    logger.info("Ensuring correct dtypes before TimeSeriesDataSet...")
    try:
        data_for_tft = df.copy()
        numeric_cols = ["time_idx", "deaths", "feature_std", "feature_lag1_std"] # Use generic names
        for col in numeric_cols: data_for_tft[col] = pd.to_numeric(data_for_tft[col], errors='coerce')
        categorical_cols = ["series_id", "month", "year"]
        for col in categorical_cols: data_for_tft[col] = data_for_tft[col].astype(str)
        if data_for_tft[numeric_cols].isnull().any().any():
             logger.warning(f"NaNs found after casting: \n{data_for_tft[numeric_cols].isnull().sum()}")
             for col in numeric_cols: data_for_tft[col].fillna(data_for_tft[col].median(), inplace=True) # Use median fill
        logger.info("Dtype check passed.")
    except Exception as e: logger.error(f"Error during dtype check: {e}", exc_info=True); return None, None, None, None

    logger.info("Setting up MONTHLY TimeSeriesDataSet for TFT (Using Default Normalization)...")
    try:
        # Use the generic standardized feature names here
        training_dataset = TimeSeriesDataSet(
            data_for_tft[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx", target="deaths", group_ids=["series_id"],
            max_encoder_length=encoder_length, max_prediction_length=pred_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=["year"],
            time_varying_unknown_reals=[ "deaths", "feature_std", "feature_lag1_std" ], # Use generic names
            # Omit target_normalizer and scalers to use defaults
            add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=True,
            # Use NaNLabelEncoder for categoricals that might have NAs after merge/cast (though we try to prevent this)
            categorical_encoders={"year": NaNLabelEncoder(add_nan=True), "month": NaNLabelEncoder(add_nan=True)}
        )
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data_for_tft, predict=True, stop_randomization=True)
        effective_batch_size = max(1, min(batch_size, len(training_dataset) // 2 if len(training_dataset) > 1 else 1))
        logger.info(f"Using effective batch size: {effective_batch_size}")
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=effective_batch_size, num_workers=0, persistent_workers=False, pin_memory=(DEVICE.type == 'cuda'))
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=effective_batch_size * 2, num_workers=0, persistent_workers=False, pin_memory=(DEVICE.type == 'cuda'))
        if len(train_dataloader) == 0 or len(val_dataloader) == 0: logger.error("Empty dataloader(s)."); return None, None, None, None
    except Exception as e:
        logger.error(f"Error creating TimeSeriesDataSet/Dataloaders: {e}", exc_info=True); logger.error(f"Data info:\n{data_for_tft.info()}"); return None, None, None, None

    logger.info("Configuring TemporalFusionTransformer model...")
    loss_metric = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    try:
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset, learning_rate=lr, hidden_size=hidden_size, attention_head_size=attn_heads,
            dropout=dropout, hidden_continuous_size=hidden_cont_size, loss=loss_metric, log_interval=10,
            optimizer="adam", reduce_on_plateau_patience=5,
        )
        logger.info(f"TFT model parameters: {tft.size()/1e6:.1f} million")
    except Exception as e: logger.error(f"Error initializing TFT: {e}", exc_info=True); return None, None, val_dataloader, validation_dataset

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accelerator, devices = ('cpu', 1) # Hardcoded CPU
    logger.info(f"Configuring Trainer (Accelerator: {accelerator}, Devices: {devices})...")
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name="tft_fear_model")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, gradient_clip_val=clip_val, callbacks=[lr_monitor, early_stop_callback], logger=tb_logger, enable_progress_bar=True)

    logger.info("Starting TFT model training (Monthly)...")
    start_train_time = time.time()
    try:
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        logger.info(f"TFT training finished in {(time.time() - start_train_time)/60:.2f} minutes.")
        best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from checkpoint: {best_model_path}")
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=DEVICE)
            return best_tft, trainer, val_dataloader, validation_dataset
        else: logger.warning("Best checkpoint not found/saved. Returning last model state."); last_model = trainer.model if hasattr(trainer, 'model') else tft; last_model.to(DEVICE); return last_model, trainer, val_dataloader, validation_dataset
    except Exception as e: logger.error(f"Error during TFT fitting: {e}", exc_info=True); return None, trainer, val_dataloader, validation_dataset

# --- evaluate_model function (Corrected Signature) ---
def evaluate_model(model: TemporalFusionTransformer, dataloader: torch.utils.data.DataLoader, dataset: TimeSeriesDataSet, plot_dir: str) -> Dict[str, float]:
    """Evaluates TFT model, returns metrics, saves plots."""
    logger.info("Evaluating model performance on validation set...")
    results = {}
    if model is None or dataloader is None or len(dataloader) == 0: logger.error("Model/Dataloader missing for eval."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
    try: eval_device = next(model.parameters()).device
    except Exception: eval_device = torch.device(DEVICE); model.to(eval_device)
    logger.info(f"Evaluation device: {eval_device}")

    actuals_list, predictions_list = [], []
    with torch.no_grad():
         for x, y in iter(dataloader):
             x = {k: v.to(eval_device) for k, v in x.items() if isinstance(v, torch.Tensor)}; target = y[0].to(eval_device)
             preds = model(x)["prediction"]; predictions_list.append(preds.cpu()); actuals_list.append(target.cpu())
    if not predictions_list: logger.error("No predictions collected."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    actuals_tensor = torch.cat(actuals_list, dim=0); predictions_tensor = torch.cat(predictions_list, dim=0)
    preds_median_flat = predictions_tensor[:, :, 1].flatten().numpy(); actuals_flat = actuals_tensor.flatten().numpy()
    min_len = min(len(actuals_flat), len(preds_median_flat))
    if len(actuals_flat) != len(preds_median_flat): logger.warning(f"Eval length mismatch: {len(actuals_flat)} vs {len(preds_median_flat)}. Truncate."); preds_median_flat=preds_median_flat[:min_len]; actuals_flat=actuals_flat[:min_len]
    preds_median_flat = np.maximum(0, preds_median_flat)

    val_mae = mean_absolute_error(actuals_flat, preds_median_flat); val_mse = mean_squared_error(actuals_flat, preds_median_flat)
    denominator = (np.abs(actuals_flat) + np.abs(preds_median_flat)) / 2.0; val_smape = np.mean(np.abs(preds_median_flat - actuals_flat) / np.where(denominator == 0, 1, denominator)) * 100
    results = {"MAE": val_mae, "MSE": val_mse, "SMAPE": val_smape}
    logger.info(f"[Validation Metrics] MAE={val_mae:.3f} MSE={val_mse:.3f} SMAPE={val_smape:.3f}%")

    # --- Plotting ---
    try:
        prediction_output = model.predict(dataloader, mode="raw", return_x=True, return_index=True)

        # Unpack assuming [predictions_dict, x_dict, index_df] structure
        # Adjust indices if your version returns a different order or tuple
        if isinstance(prediction_output, (list, tuple)) and len(prediction_output) == 3:
            raw_preds_dict = prediction_output[0] # Dictionary containing predictions
            x_dict = prediction_output[1]         # Dictionary containing input data
            index_df = prediction_output[2]       # DataFrame containing index info
        else:
            # Fallback or handle other potential return types if needed
            logger.error(f"Unexpected return type from model.predict: {type(prediction_output)}. Cannot unpack for plotting.")
            # Return metrics only if plotting fails due to unexpected format
            return results

        # Access predictions from the dictionary
        p10_all = raw_preds_dict['prediction'][:, :, 0].flatten().cpu().numpy()
        p50_all = raw_preds_dict['prediction'][:, :, 1].flatten().cpu().numpy()
        p90_all = raw_preds_dict['prediction'][:, :, 2].flatten().cpu().numpy()
        actuals_all_list = [y[0].cpu() for _, y in iter(dataloader)]; actuals_all = torch.cat(actuals_all_list).flatten().numpy()
        min_len_final = min(len(p50_all), len(actuals_all)); p10_all=p10_all[:min_len_final]; p50_all=p50_all[:min_len_final]; p90_all=p90_all[:min_len_final]; actuals_all=actuals_all[:min_len_final]
        time_idx = index_df["time_idx"].values[:min_len_final]
        idx_order = np.argsort(time_idx); time_idx_sorted=time_idx[idx_order]; act_sorted=actuals_all[idx_order]; p10_sorted=p10_all[idx_order]; p50_sorted=p50_all[idx_order]; p90_sorted=p90_all[idx_order]

        try: min_date_str = dataset.data["raw"]["month_date"].min().strftime('%Y-%m'); x_label = f"Time Index (Months since {min_date_str})"
        except Exception: x_label = "Time Index"

        plt.figure(figsize=(15, 7)); plt.plot(time_idx_sorted, act_sorted, label="Actual Deaths", marker='.', linestyle='-', alpha=0.6, color='black', markersize=4)
        plt.plot(time_idx_sorted, p50_sorted, label="Predicted Median (p50)", linestyle='--', alpha=0.8, color='tab:orange', linewidth=1.5)
        plt.fill_between(time_idx_sorted, p10_sorted, p90_sorted, color='tab:orange', alpha=0.25, label='p10-p90 Quantiles')
        plt.title(f"TFT Aggregated Forecast vs Actuals (Validation Set - Monthly)\nMAE={val_mae:.2f}, SMAPE={val_smape:.2f}%", fontsize=14); plt.xlabel(x_label, fontsize=12); plt.ylabel("Monthly Deaths", fontsize=12)
        plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        plot_file = os.path.join(plot_dir, "tft_val_forecast_aggregated.png"); plt.savefig(plot_file); logger.info(f"Saved aggregated forecast plot to {plot_file}"); plt.close()

        residuals = act_sorted - p50_sorted
        plt.figure(figsize=(10, 6)); plt.scatter(p50_sorted, residuals, alpha=0.3, s=15, color='tab:blue', edgecolors='w', linewidth=0.5)
        plt.axhline(0, color='red', linestyle='--', linewidth=1); plt.title('Residual Plot (Actuals - Median Predictions)', fontsize=14)
        plt.xlabel('Predicted Median Deaths', fontsize=12); plt.ylabel('Residuals', fontsize=12); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        save_path_res = os.path.join(plot_dir, "residuals_monthly_plot.png"); plt.savefig(save_path_res); logger.info(f"Saved residual plot to {save_path_res}"); plt.close()
    except Exception as e: logger.warning(f"Evaluation plotting failed: {e}", exc_info=True)
    finally: plt.close('all')
    return results

# -----------------------------------------------------------------------------
# 6. Enhanced Plotting Functions  
# -----------------------------------------------------------------------------
def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str, ylabel: str, filename: str, plot_dir: str):
    logger.info(f"Generating plot: {title}")
    if time_col not in df.columns or value_col not in df.columns: logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        plt.figure(figsize=(15, 5)); plt.plot(df[time_col], df[value_col], marker='.', linestyle='-', markersize=2, alpha=0.7, linewidth=1)
        plt.title(title); plt.xlabel(time_col.replace('_', ' ').title()); plt.ylabel(ylabel); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

def plot_dual_axis(df: pd.DataFrame, time_col: str, col1: str, col2: str, label1: str, label2: str, title: str, filename: str, plot_dir: str):
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [time_col, col1, col2]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    fig, ax1 = plt.subplots(figsize=(15, 6))
    try:
        time_data = df[time_col]; x_label = time_col.replace('_', ' ').title()
        if pd.api.types.is_datetime64_any_dtype(time_data): pass
        else: min_date_str = df['month_date'].min().strftime('%Y-%m'); x_label = f"Time Index (Months since {min_date_str})"
        color1 = 'tab:blue'; ax1.set_xlabel(x_label); ax1.set_ylabel(label1, color=color1)
        line1 = ax1.plot(time_data, df[col1], color=color1, label=label1, alpha=0.8, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax2 = ax1.twinx(); color2 = 'tab:red'; ax2.set_ylabel(label2, color=color2)
        line2 = ax2.plot(time_data, df[col2], color=color2, label=label2, linestyle='--', alpha=0.8, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color2); fig.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.title(title);
        lines = line1 + line2; labels = [l.get_label() for l in lines]; ax1.legend(lines, labels, loc='upper left')
        plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close(fig)

def plot_scatter_fear_vs_deaths(df: pd.DataFrame, fear_col: str, death_col: str, title: str, filename: str, plot_dir: str):
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [fear_col, death_col]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        plt.figure(figsize=(8, 8)); sns.scatterplot(data=df, x=fear_col, y=death_col, alpha=0.3, s=15, edgecolor=None)
        corr = df[fear_col].corr(df[death_col]); plt.title(f"{title}\n(Correlation: {corr:.2f})")
        plt.xlabel(fear_col.replace('_', ' ').title()); plt.ylabel(death_col.replace('_', ' ').title())
        plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

def plot_monthly_boxplot(df: pd.DataFrame, column: str, title: str, filename: str, plot_dir: str):
    logger.info(f"Generating plot: {title}")
    if 'month' not in df.columns or column not in df.columns: logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        df['month_num'] = df['month'].astype(int); month_order = [str(i) for i in range(1, 13)]
        plt.figure(figsize=(12, 6)); sns.boxplot(x='month', y=column, data=df, order=month_order, showfliers=False, palette="viridis")
        plt.title(title); plt.xlabel("Month"); plt.ylabel(column.replace('_', ' ').title()); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

# -----------------------------------------------------------------------------
# 7. Interpretation & Granger Causality  
# -----------------------------------------------------------------------------
def interpret_tft(model: TemporalFusionTransformer, val_dataloader: torch.utils.data.DataLoader, plot_dir: str):
    logger.info("Calculating TFT feature importance...")
    if model is None or val_dataloader is None or len(val_dataloader) == 0: logger.warning("Model/Dataloader missing, skipping interpretation."); return
    try:
        interpret_device = "cpu"; model.to(interpret_device); logger.info(f"Running interpretation on: {interpret_device}")
        prediction_output = model.predict(val_dataloader, mode="raw", return_x=True)

        # Unpack assuming [predictions_dict, x_dict] structure
        if isinstance(prediction_output, (list, tuple)) and len(prediction_output) >= 1:
            # We mainly need the predictions dict for interpret_output
            raw_predictions_dict = prediction_output[0]
            # Ensure the prediction tensors within the dict are on the CPU
            raw_predictions_dict = {
                k: v.to(interpret_device) if isinstance(v, torch.Tensor) else v
                for k, v in raw_predictions_dict.items()
            }
        else:
            logger.error(f"Unexpected return type from model.predict: {type(prediction_output)}. Cannot unpack for interpretation.")
            return # Stop interpretation

        # Calculate interpretation using the unpacked dictionary
        interpretation = model.interpret_output(raw_predictions_dict, reduction="mean") # Pass the dict here
        logger.info("Plotting TFT interpretation...")
        fig_imp = model.plot_interpretation(interpretation, plot_type="variable_importance"); fig_imp.suptitle("TFT Feature Importance (Monthly)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_path_imp = os.path.join(plot_dir, "tft_interpretation_importance.png"); fig_imp.savefig(save_path_imp); logger.info(f"Saved interpretation plot to {save_path_imp}"); plt.close(fig_imp)
    except ImportError: logger.warning("matplotlib issue. Skip interpretation plot.")
    except Exception as e: logger.error(f"Error during interpretation: {e}", exc_info=True)
    finally: model.to(DEVICE); plt.close('all')

def run_granger_causality(df: pd.DataFrame, var1: str, var2: str, max_lag: int = 12):
    logger.info(f"Running Granger Causality test: '{var1}' -> '{var2}'? (Max lag: {max_lag})")
    if var1 not in df.columns or var2 not in df.columns: logger.error(f"Granger cols missing. Skip."); return
    data = df[[var1, var2]].copy()
    if data.isnull().values.any(): logger.warning(f"NaNs found in Granger data: {data.isnull().sum()}. Drop."); data.dropna(inplace=True)
    if data.shape[0] < max_lag + 5: logger.error(f"Not enough data ({data.shape[0]}) for Granger test after dropna. Skip."); return
    try: data_diff = data.diff().dropna()
    except Exception as e: logger.error(f"Granger differencing error: {e}. Skip."); return
    if data_diff.shape[0] < max_lag + 5: logger.error(f"Not enough data ({data_diff.shape[0]}) after differencing. Skip."); return
    logger.info("Using first differences for Granger test.")
    try:
        gc_result = grangercausalitytests(data_diff[[var2, var1]], maxlag=max_lag, verbose=False)
        significant_lags = [lag for lag in range(1, max_lag + 1) if gc_result[lag][0]['ssr_ftest'][1] < 0.05]
        if significant_lags: logger.info(f" Granger Result ('{var1}' -> '{var2}'): Significant at lags: {significant_lags} (p < 0.05).")
        else: logger.info(f" Granger Result ('{var1}' -> '{var2}'): Not significant up to lag {max_lag} (p >= 0.05).")
    except Exception as e: logger.error(f"Granger test error: {e}", exc_info=True)

# -----------------------------------------------------------------------------
# 8. MAIN Execution Logic
# -----------------------------------------------------------------------------
def main():
    """Main script execution flow."""
    logger.info("--- Starting Historical Analysis Script (Monthly Focus, Yearly Fear) ---")
    script_start_time = time.time()
    final_df = None; best_model = None; trainer = None; val_dl = None; val_ds = None

    try:
        os.makedirs(PLOT_DIR, exist_ok=True); logger.info(f"Plots directory: {PLOT_DIR}")

        # --- Step 1: Parse ECCO Data ---
        logger.info("--- Step 1: Parsing & Filtering ECCO Data ---")
        ecco_df_raw = parse_ecco_tei_files(ecco_dir=ECCO_DIR, start_year=START_YEAR, end_year=END_YEAR, relevant_genres=RELEVANT_GENRES)
        if not isinstance(ecco_df_raw, pd.DataFrame) or ecco_df_raw.empty: raise ValueError("Step 1 Failed: No ECCO documents found matching CRITERIA (Check XML XPaths/Filters).")

        # --- Step 2: Preprocess Text ---
        logger.info("--- Step 2: Preprocessing ECCO Text ---")
        USE_SYMSPELL_CORRECTION = False # Keep False unless you have a good historical dict
        ecco_df_processed = preprocess_text_dataframe(ecco_df_raw, text_col='text', use_symspell=USE_SYMSPELL_CORRECTION)
        if ecco_df_processed.empty: raise ValueError("Step 2 Failed: Text preprocessing resulted in empty DataFrame.")
        del ecco_df_raw # Free memory

        # --- Step 3: Calculate Fear Scores ---
        logger.info("--- Step 3: Calculating Fear Scores using MacBERTh ---")
        ecco_df_scored = calculate_fear_scores_dataframe(ecco_df_processed, text_col='processed_text', batch_size=BATCH_SIZE // 2)
        if 'fear_score' not in ecco_df_scored.columns: raise ValueError("Step 3 Failed: Fear scoring did not produce results.")
        del ecco_df_processed # Free memory

        # --- Step 4: Load Monthly Mortality ---
        logger.info("--- Step 4: Loading and Processing Monthly Mortality Data ---")
        monthly_mortality_df = load_and_aggregate_monthly_mortality(file_path=COUNTS_FILE, start_year=START_YEAR, end_year=END_YEAR)
        if monthly_mortality_df.empty: raise ValueError("Step 4 Failed: No monthly mortality data found.")

        # --- Step 5: Aggregate Sentiment & Merge ---
        logger.info(f"--- Step 5: Aggregating Yearly Sentiment ('{AGGREGATION_METHOD}') and Merging ---")
        final_df = aggregate_yearly_sentiment_and_merge_monthly(ecco_df_scored, monthly_mortality_df, agg_method=AGGREGATION_METHOD, fear_thresh=FEAR_THRESHOLD)
        if final_df.empty: raise ValueError("Step 5 Failed: Merging resulted in empty DataFrame.")
        del ecco_df_scored; del monthly_mortality_df # Free memory

        # Save the final data used for modeling
        try: final_df.to_csv(os.path.join(PLOT_DIR, f"final_monthly_data_fear_{AGGREGATION_METHOD}.csv"), index=False); logger.info("Saved final monthly data.")
        except Exception as e: logger.warning(f"Could not save final data CSV: {e}")

        # --- Step 5.5: Generate Data Plots ---
        logger.info("--- Step 5.5: Generating Data Exploration Plots ---")
        # Define the column names based on the chosen aggregation method
        raw_fear_col = f'fear_score_yearly_{AGGREGATION_METHOD}'
        lag1_fear_col = f'{raw_fear_col}_lag1'
        std_fear_col = 'feature_std' # Generic standardized name
        lag1_std_fear_col = 'feature_lag1_std' # Generic standardized name

        if final_df is not None and not final_df.empty:
            plot_time_series(final_df, 'month_date', 'deaths', 'Monthly Deaths Over Time', 'Total Deaths', 'deaths_monthly_timeseries.png', PLOT_DIR)
            plot_time_series(final_df, 'month_date', raw_fear_col, f'Yearly Fear ({AGGREGATION_METHOD.title()}) (Mapped to Month)', 'Fear Score', f'fear_yearly_{AGGREGATION_METHOD}_timeseries.png', PLOT_DIR)
            plot_time_series(final_df, 'month_date', lag1_fear_col, f'Lagged (1Y) Yearly Fear ({AGGREGATION_METHOD.title()}) (Mapped to Month)', 'Fear Score', f'fear_yearly_{AGGREGATION_METHOD}_lag1_timeseries.png', PLOT_DIR)
            plot_dual_axis(final_df, 'month_date', 'deaths', std_fear_col, 'Monthly Deaths', f'Std. Yearly Fear ({AGGREGATION_METHOD.title()})', f'Monthly Deaths vs Std. Current Year Fear ({AGGREGATION_METHOD})', f'deaths_vs_fear_current_{AGGREGATION_METHOD}_std_monthly_dual_axis.png', PLOT_DIR)
            plot_dual_axis(final_df, 'month_date', 'deaths', lag1_std_fear_col, 'Monthly Deaths', f'Std. Yearly Fear ({AGGREGATION_METHOD.title()}, Lagged 1Y)', f'Monthly Deaths vs Std. Previous Year Fear ({AGGREGATION_METHOD})', f'deaths_vs_fear_lag1_{AGGREGATION_METHOD}_std_monthly_dual_axis.png', PLOT_DIR)
            plot_scatter_fear_vs_deaths(final_df, lag1_fear_col, 'deaths', f'Monthly Deaths vs Previous Year Fear ({AGGREGATION_METHOD.title()})', f'scatter_deaths_vs_fear_{AGGREGATION_METHOD}_lag1.png', PLOT_DIR)
            plot_scatter_fear_vs_deaths(final_df, raw_fear_col, 'deaths', f'Monthly Deaths vs Current Year Fear ({AGGREGATION_METHOD.title()})', f'scatter_deaths_vs_fear_{AGGREGATION_METHOD}_current.png', PLOT_DIR)
            plot_monthly_boxplot(final_df, 'deaths', 'Monthly Distribution of Deaths', 'boxplot_deaths_monthly.png', PLOT_DIR)
        else: logger.warning("final_df empty, skipping data plots.")

        # === Step 6: Train and Evaluate TFT Model ===
        logger.info("--- Step 6: Training and Evaluating TFT Model (Monthly) ---")
        if final_df is not None and not final_df.empty:
            best_model, trainer, val_dl, val_ds = train_tft_model(df=final_df) # Using default hyperparameters from config section
            if best_model and val_dl and val_ds:
                logger.info("--- Step 6.5: Evaluating Best TFT Model ---")
                # Pass dataset object needed for evaluation plotting metadata
                eval_metrics = evaluate_model(best_model, val_dl, val_ds, plot_dir=PLOT_DIR) # Corrected call
                logger.info(f"Final Validation Metrics: {eval_metrics}")
            else: logger.error("TFT Model training failed or did not return model/dataloader.")
        else: logger.error("Final DataFrame empty, cannot train TFT model.")

        # === Step 7: Interpretation and Granger Causality ===
        logger.info("--- Step 7: Model Interpretation & Granger Causality ---")
        if final_df is not None and not final_df.empty:
            if best_model and val_dl: interpret_tft(best_model, val_dl, PLOT_DIR)
            else: logger.warning("Skipping TFT interpretation.")
            # Run Granger on the chosen raw aggregation features
            logger.info("--- Running Granger Causality Tests ---")
            run_granger_causality(final_df, raw_fear_col, 'deaths', max_lag=12)
            run_granger_causality(final_df, lag1_fear_col, 'deaths', max_lag=12)
            run_granger_causality(final_df, 'deaths', raw_fear_col, max_lag=12) # Reverse direction
        else: logger.warning("Final DataFrame empty, skipping interpretation & Granger.")

    except FileNotFoundError as e: logger.error(f"Data file not found: {e}.")
    except ValueError as e: logger.error(f"Data processing/config error: {e}", exc_info=True)
    except ImportError as e: logger.error(f"Missing library: {e}.")
    except Exception as e: logger.error(f"Unexpected script error: {e}", exc_info=True)

    script_end_time = time.time()
    logger.info(f"--- Script finished in {(script_end_time - script_start_time):.2f} seconds ({(script_end_time - script_start_time)/60:.2f} minutes) ---")

# -----------------------------------------------------------------------------
# Run Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()