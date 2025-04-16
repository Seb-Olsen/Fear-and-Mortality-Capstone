#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical Analysis Script - Old Bailey Sessions Papers & Weekly Bills of Mortality
Analyzes correlation between WEEKLY fear sentiment (derived using MacBERTh embeddings
from Old Bailey) and WEEKLY mortality trends using TFT forecasting.

*** TARGET ENVIRONMENT: Python 3.10+, pytorch-forecasting (latest compatible),
                       pytorch-lightning (latest compatible), transformers, sentence-transformers,
                       torch, nltk, pandas, numpy, matplotlib, seaborn, joblib, symspellpy, statsmodels ***
"""

import os
import logging
import re
import warnings
from datetime import datetime, timedelta # Added timedelta
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
warnings.filterwarnings("ignore", ".*does not have valid feature names*")

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
        try: nltk.data.find(path); logger.debug(f"NLTK data '{package_id}' found."); package_found = True
        except LookupError: package_found = False
        except Exception as e: logger.warning(f"NLTK check failed: {e}"); package_found = False
        if not package_found:
            logger.info(f"Downloading NLTK package: {package_id}")
            try:
                force_dl = (package_id == 'punkt')
                if not downloader.download(package_id, quiet=True, force=force_dl):
                    try: nltk.data.find(path); logger.info(f"NLTK '{package_id}' found after check.")
                    except LookupError: raise RuntimeError(f"Failed download/locate: {package_id}")
                else: logger.info(f"NLTK '{package_id}' downloaded.")
            except Exception as e: logger.error(f"NLTK download error: {e}", exc_info=True); raise
download_nltk_data()

# ---------------------
# Other Imports
# ---------------------
from symspellpy.symspellpy import SymSpell, Verbosity
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import grangercausalitytests
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, EncoderNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from joblib import Memory

# ---------------------
# Configuration (UPDATED FOR OLD BAILEY & WEEKLY ANALYSIS)
# ---------------------
OLD_BAILEY_DIR = '/Users/sebo/Desktop/AUC/Semester 6/Capstone/Programming/oldbailey/sessionsPapers/' # *** YOUR PATH ***
COUNTS_FILE = '/Users/sebo/Desktop/AUC/Semester 6/Capstone/Programming/WeeklyBillsMortality1644to1849/counts.txt' # *** YOUR PATH ***
HISTORICAL_DICT_PATH = "historical_dict.txt" # Optional
START_YEAR = 1678 # Old Bailey data starts around here
END_YEAR = 1849   # Align end year with previous analysis for now
INFECTIOUS_DISEASE_TYPES = None # Use None for total mortality (excluding christened)

# --- Sentiment Analysis Config ---
MACBERTH_MODEL_NAME = 'emanjavacas/MacBERTh'
FEAR_WORDS = [ # Refined list including crime context
    "fear", "afraid", "scared", "terror", "dread", "panic", "anxiety", "worry",
    "horror", "phobia", "fright", "alarm", "apprehension", "nervousness",
    "trembling", "timidity", "consternation", "distress", "unease", "danger",
    "pestilence", "plague", "contagion", "infection", "epidemic", "pox",
    "sickness", "disease", "illness", "malady", "distemper",
    "dying", "death", "mortality", "fatal", "corpse", "grave", "burial",
    "suffering", "agony", "misery", "calamity", "crisis",
    "murder", "robbery", "theft", "violence", "crime", "villain", "malefactor",
    "guilty", "hanged", "executed", "sentence", "prison", "newgate", "punishment"
]
FEAR_WORDS = list(set(FEAR_WORDS)) # Unique words

# --- Aggregation Method (Now WEEKLY) ---
AGGREGATION_METHOD = 'mean' # 'mean', 'max', or 'proportion'
FEAR_THRESHOLD = 0.70 # Example threshold if using 'proportion'

# --- TFT Parameters (Weekly) ---
MAX_EPOCHS = 50
BATCH_SIZE = 64 # Can often increase slightly for weekly data vs monthly
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.2
HIDDEN_CONTINUOUS_SIZE = 16
WEEKLY_MAX_ENCODER_LENGTH = 52 * 1 # Use 1 year (52 weeks) of history
WEEKLY_MAX_PREDICTION_LENGTH = 8  # Predict 8 weeks ahead
GRADIENT_CLIP_VAL = 0.15

# --- Plotting Directory ---
PLOT_DIR = "plots_weekly_oldbailey" # New plot dir
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

# --- Device Setup ---
logger.info("Forcing CPU due to potential MPS compatibility issues or preference.")
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# --- Caching Setup ---
cache_dir = "cache_dir_weekly_oldbailey" # New cache dir
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)


# -----------------------------------------------------------------------------
# 1. Text Normalization & Preprocessing Helpers (Unchanged)
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
# 2. SymSpell Setup & Correction (Optional, Unchanged)
# -----------------------------------------------------------------------------
@memory.cache
def setup_symspell(dictionary_path=HISTORICAL_DICT_PATH, max_edit_distance=1):
    if not os.path.exists(dictionary_path): logger.warning(f"SymSpell dict NF: {dictionary_path}. Skip."); return None
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        try: loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
        except UnicodeDecodeError: logger.warning("UTF-8 failed, try latin-1."); loaded = sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='latin-1')
        if not loaded: logger.error(f"Failed load SymSpell dict: {dictionary_path}"); return None
        logger.info(f"SymSpell loaded (max_edit={max_edit_distance})."); return sym_spell
    except Exception as e: logger.error(f"SymSpell setup error: {e}", exc_info=True); return None
sym_spell_global = setup_symspell()

def correct_ocr_spelling(text: str, sym_spell: Optional[SymSpell]) -> str:
    if not sym_spell or not isinstance(text, str) or not text.strip(): return text
    words = text.split(); corrected_words = []
    for word in words:
        clean_word = word.strip('.,!?;:"()[]')
        if not clean_word or not clean_word.isalpha(): corrected_words.append(word); continue
        suggestions = sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=sym_spell.max_dictionary_edit_distance, include_unknown=True)
        if suggestions:
            best_suggestion = suggestions[0].term; apply_correction = best_suggestion.lower() != clean_word.lower()
            if apply_correction:
                if word.istitle() and len(word) > 1: corrected_words.append(best_suggestion.capitalize())
                elif word.isupper() and len(word) > 1: corrected_words.append(best_suggestion.upper())
                else: corrected_words.append(best_suggestion)
            else: corrected_words.append(word)
        else: corrected_words.append(word)
    return " ".join(corrected_words)

# -----------------------------------------------------------------------------
# 3. Core Text Preprocessing Function (Unchanged)
# -----------------------------------------------------------------------------
def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set, sym_spell: Optional[SymSpell] = None, use_symspell: bool = False) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = normalize_historical_text(text)
    if use_symspell and sym_spell: text = correct_ocr_spelling(text, sym_spell)
    text_cleaned = re.sub(r"[^\w\s]", " ", text); text_cleaned = re.sub(r"\d+", "", text_cleaned)
    text_cleaned = text_cleaned.lower(); text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    if not text_cleaned: return ""
    try: tokens = nltk.word_tokenize(text_cleaned)
    except Exception as e: logger.warning(f"Tokenization failed: {e}"); return ""
    if not tokens: return ""
    try: tagged_tokens = nltk.pos_tag(tokens)
    except Exception as e: logger.warning(f"POS Tagging failed: {e}. Default noun."); tagged_tokens = [(t, 'NN') for t in tokens]
    processed_tokens = []
    for word, tag in tagged_tokens:
        if word not in stop_words and len(word) > 1:
            try: lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)); processed_tokens.append(lemma)
            except Exception as e: logger.warning(f"Lemmatization failed: {word}/{tag}: {e}"); processed_tokens.append(word)
    return " ".join(processed_tokens)

# -----------------------------------------------------------------------------
# 4. Cached Data Loading and Processing Functions (UPDATED FOR WEEKLY)
# -----------------------------------------------------------------------------

# === Old Bailey Parsing (NEW) ===
@memory.cache
def parse_old_bailey_papers(ob_dir: str = OLD_BAILEY_DIR, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    (Cached) Parses Old Bailey Sessions Papers XML files (TEI.2 format).
    Extracts the session start date from <interp type="date" value="YYYYMMDD">
    or <div0 id="YYYYMMDD">. Maps date to the start of the ISO week (Monday).
    Extracts all text within <p> tags inside the <body>.
    Returns DataFrame with ['week_date', 'text', 'doc_id'].
    """
    records = []
    logger.info(f"Starting Old Bailey Sessions Papers parsing (Years: {start_year}-{end_year})...")
    file_count = 0; processed_count = 0; skipped_date = 0; skipped_text = 0; parse_errors = 0; date_parse_attempts = 0

    for rootdir, _, files in os.walk(ob_dir):
        for fname in files:
            # Assume files might or might not have .xml extension
            if not (fname.endswith('.xml') or re.match(r'^\d{8}$', fname) or '.' not in fname): # Allow extensionless IDs
                 continue
            file_count += 1
            if file_count % 500 == 0: logger.info(f" Scanning file {file_count}...")

            fpath = os.path.join(rootdir, fname)
            # Use filename without extension as potential doc_id
            base_fname = os.path.splitext(fname)[0]

            try:
                tree = ET.parse(fpath)
                root = tree.getroot() # Should be <TEI.2>

                # --- Extract Date ---
                # Priority 1: <interp type="date" value="YYYYMMDD"> inside <div0>
                # Priority 2: <div0 id="YYYYMMDD">
                session_div = root.find('.//div0[@type="sessionsPaper"]')
                session_date_str = None
                session_date = None

                if session_div is not None:
                    interp_date_node = session_div.find('.//interp[@type="date"]')
                    if interp_date_node is not None and 'value' in interp_date_node.attrib:
                        session_date_str = interp_date_node.attrib['value']
                        date_parse_attempts += 1
                    elif 'id' in session_div.attrib: # Fallback to div0 id
                        session_date_str = session_div.attrib['id']
                        date_parse_attempts += 1

                if session_date_str and re.match(r'^\d{8}$', session_date_str):
                    try:
                        session_date = datetime.strptime(session_date_str, '%Y%m%d')
                        # Filter by year
                        if not (start_year <= session_date.year <= end_year):
                            skipped_date += 1; continue
                    except ValueError:
                        logger.warning(f"Date parse failed '{session_date_str}' in {fname}. Skip."); skipped_date += 1; continue
                else:
                    logger.debug(f"No valid date found in {fname}. Skip."); skipped_date += 1; continue

                # --- Map Session Date to Start of ISO Week (Monday) ---
                iso_year, iso_week, _ = session_date.isocalendar()
                try:
                    week_start_date = datetime.fromisocalendar(iso_year, iso_week, 1)
                except ValueError: logger.warning(f"Week start date fail for {session_date_str} in {fname}. Skip."); skipped_date += 1; continue

                # --- Extract Text ---
                body = root.find('.//body')
                if body is None: logger.warning(f"No body found in {fname}"); skipped_text += 1; continue
                all_paragraphs = body.findall('.//p'); full_text_parts = []
                for p_node in all_paragraphs:
                    node_text = ' '.join(t.strip() for t in p_node.itertext() if t and t.strip())
                    node_text_clean = re.sub(r'\s+', ' ', node_text).strip()
                    if node_text_clean: full_text_parts.append(node_text_clean)
                if not full_text_parts: logger.debug(f"No <p> text found in {fname}"); skipped_text += 1; continue
                combined_text = ' '.join(full_text_parts)

                # Use the extracted date string (YYYYMMDD) as a more reliable doc_id
                doc_id = session_date_str if session_date_str else base_fname

                records.append({'week_date': week_start_date, 'text': combined_text, 'doc_id': doc_id})
                processed_count += 1
                if processed_count % 500 == 0: logger.info(f" Found {processed_count} valid Old Bailey records...")

            except ET.ParseError as e: logger.warning(f"XML Parse Error {fname}: {e}"); parse_errors += 1
            except Exception as e: logger.warning(f"General Error {fname}: {e}", exc_info=False); parse_errors += 1

    logger.info(f"Finished Old Bailey parsing. Files scanned: {file_count}")
    if processed_count == 0: logger.error("CRITICAL: No Old Bailey records processed. Check directory, file format, or date extraction logic.")
    else: logger.info(f" Processed {processed_count} valid records.")
    logger.info(f" Skipped: {skipped_date} (date issue), {skipped_text} (no text).")
    logger.info(f" Errors during parsing: {parse_errors}. Date parse attempts: {date_parse_attempts}")
    if not records: return pd.DataFrame(columns=["week_date", "text", "doc_id"])

    # Define the minimum representable date in pandas ns precision
    # (approximately 1677-09-21, let's use 1678-01-01 to be safe)
    min_pandas_date = datetime(1678, 1, 1)
    # You could also adjust START_YEAR in config to 1678 if acceptable

    # Filter records *before* creating the DataFrame
    original_record_count = len(records)
    filtered_records = [r for r in records if r['week_date'] >= min_pandas_date]
    filtered_count = len(filtered_records)
    if filtered_count < original_record_count:
        logger.warning(f"Filtered out {original_record_count - filtered_count} Old Bailey records with dates before {min_pandas_date.strftime('%Y-%m-%d')} due to Pandas timestamp limitations.")

    if not filtered_records:
         logger.error("No Old Bailey records remain after filtering for valid Pandas date range.")
         return pd.DataFrame(columns=["week_date", "text", "doc_id"])

    # Now create the DataFrame only with valid dates
    df = pd.DataFrame(filtered_records)
    df['text'] = df['text'].astype(str)
    # This conversion should now succeed
    df['week_date'] = pd.to_datetime(df['week_date'])

    logger.info(f"Old Bailey DataFrame prepared: {df.shape[0]} records (after date filtering). Date Range: {df['week_date'].min():%Y-%m-%d} to {df['week_date'].max():%Y-%m-%d}")
    return df


# === Mortality Loading (WEEKLY - UPDATED) ===
def parse_bill_weekID_to_weekly(week_str: str) -> Optional[datetime]:
    """Parses Bills of Mortality weekID (YYYY/WW) to a datetime object for the START of the ISO WEEK (Monday)."""
    try:
        year_str, week_ = week_str.split("/")
        year = int(year_str); week = int(week_)
        # Use START_YEAR and END_YEAR from config
        if not (START_YEAR <= year <= END_YEAR) or not (1 <= week <= 53): return None
        # Calculate Monday of the given ISO year and week
        return datetime.fromisocalendar(year, week, 1)
    except ValueError:
        # Handle week 53 issue for years that don't have it
        if week == 53:
            try: return datetime.fromisocalendar(year, 52, 1) # Fallback to week 52
            except ValueError: logger.debug(f"Invalid week 52/53 for year {year_str}."); return None
        else: logger.debug(f"Cannot parse week {week_} for year {year_str}."); return None
    except Exception as e: logger.debug(f"Error parsing bill weekID '{week_str}' to weekly: {e}"); return None

@memory.cache
def load_and_aggregate_weekly_mortality(file_path: str = COUNTS_FILE, disease_types: Optional[List[str]] = INFECTIOUS_DISEASE_TYPES, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    (Cached) Loads mortality counts, aggregates to WEEKLY totals (start of week - Monday),
    filtering by year range and optionally by disease type. Excludes 'christened'.
    Returns DataFrame with ['week_date', 'year', 'week_of_year', 'deaths'].
    """
    if not os.path.exists(file_path): raise FileNotFoundError(f"Mortality file NF: {file_path}")
    logger.info(f"Loading mortality data from {file_path}...")
    df = pd.read_csv(file_path, delimiter="|", low_memory=False, dtype={'weekID': str})
    required_cols = ["weekID", "counttype", "countn"]
    if not all(c in df.columns for c in required_cols): raise ValueError("Mortality file missing required columns")
    df["countn"] = pd.to_numeric(df["countn"], errors="coerce")
    original_len = len(df); df.dropna(subset=["countn"], inplace=True); df["countn"] = df["countn"].astype(int)
    if len(df) < original_len: logger.warning(f"Dropped {original_len - len(df)} rows non-numeric 'countn'.")

    logger.info("Parsing week IDs to week start dates (Monday)...")
    df["week_date_obj"] = df["weekID"].astype(str).apply(parse_bill_weekID_to_weekly) # Store results temporarily

    # --- Filter rows with valid dates AND within rough year range FIRST ---
    original_len = len(df)
    # Also filter out rows where week_date_obj is None (parsing failed)
    df = df.dropna(subset=["week_date_obj"]).copy()

    # *** ADD PRE-FILTER BASED ON YEAR ***
    # This requires extracting year from week_date_obj BEFORE full conversion
    # This is safe because parse_bill_weekID_to_weekly returns datetime objects
    try:
        df['temp_year'] = df['week_date_obj'].apply(lambda x: x.isocalendar().year if pd.notnull(x) else None)
        df.dropna(subset=['temp_year'], inplace=True) # Drop if year extraction failed
        df['temp_year'] = df['temp_year'].astype(int)
        df = df[(df['temp_year'] >= start_year) & (df['temp_year'] <= end_year)].copy()
        df = df.drop(columns=['temp_year']) # Remove temporary column
    except Exception as e:
         logger.error(f"Error during pre-filtering by year: {e}. Proceeding without pre-filter, might still fail.")

    dropped_rows = original_len - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with invalid weekID/date or outside year range {start_year}-{end_year}.")

    if df.empty:
         logger.warning("No rows remaining after initial date parsing and year filtering.")
         return pd.DataFrame(columns=["week_date", "year", "week_of_year", "deaths"])

    # --- NOW convert the (already filtered) column to datetime ---
    try:
        # This should now succeed as all rows have valid datetime objects within pandas range
        df['week_date'] = pd.to_datetime(df['week_date_obj'])
    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime as e:
        logger.error(f"OutOfBoundsDatetime error persist AFTER filtering: {e}. Check START_YEAR ({start_year}) config.")
        # Optionally print problematic dates:
        problematic_dates = df.loc[pd.to_datetime(df['week_date_obj'], errors='coerce').isna(), 'week_date_obj']
        logger.error(f"Problematic date objects (first 5): {problematic_dates.head().tolist()}")
        raise e # Re-raise the error as it's unexpected now

    df = df.drop(columns=['week_date_obj']) # We no longer need the temporary column

    # --- Filter by Year Range (Redundant check, but safe) ---
    df["year"] = df["week_date"].dt.isocalendar().year
    # This filter should ideally not remove anything now, but kept as safeguard
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

    # --- Continue with type filtering and aggregation ---
    df['counttype'] = df['counttype'].str.lower().str.strip()
    df = df[df["counttype"] != "christened"]
    if disease_types: df = df[df["counttype"].isin([d.lower() for d in disease_types])]
    else: logger.info("Using total mortality (excluding 'christened').")
    if df.empty: logger.warning("No records after type filter."); return pd.DataFrame(columns=["week_date", "year", "week_of_year", "deaths"])
    logger.info(f"{len(df)} weekly records after type filter.")

    logger.info("Aggregating counts per WEEK...")
    weekly_sum = df.groupby("week_date")["countn"].sum().reset_index()
    weekly_sum.rename(columns={"countn": "deaths"}, inplace=True)
    weekly_sum["deaths"] = weekly_sum["deaths"].astype(float)
    # Re-calculate year and week from the final aggregated week_date
    weekly_sum["year"] = weekly_sum["week_date"].dt.isocalendar().year
    weekly_sum["week_of_year"] = weekly_sum["week_date"].dt.isocalendar().week
    weekly_sum = weekly_sum.sort_values("week_date").reset_index(drop=True)
    logger.info(f"Mortality aggregated: {weekly_sum.shape[0]} weeks. Date Range: {weekly_sum['week_date'].min():%Y-%m-%d} to {weekly_sum['week_date'].max():%Y-%m-%d}")
    return weekly_sum[["week_date", "year", "week_of_year", "deaths"]]


# === Text Preprocessing (Unchanged Function, applied to Old Bailey Text) ===
@memory.cache
def preprocess_text_dataframe(df: pd.DataFrame, text_col: str = "text", use_symspell: bool = False) -> pd.DataFrame:
    logger.info(f"Preprocessing text column '{text_col}' (use_symspell={use_symspell})...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")
    df_copy = df.copy(); df_copy[text_col] = df_copy[text_col].astype(str).fillna('')
    lemmatizer = WordNetLemmatizer(); stop_words = set(stopwords.words("english"))
    global sym_spell_global
    total_rows = len(df_copy); processed_texts = []
    # Preprocessing can be slow, log progress less frequently
    log_interval = max(1, total_rows // 10)
    for i, text in enumerate(df_copy[text_col]):
         if (i + 1) % log_interval == 0: logger.info(f" Preprocessing text {i+1}/{total_rows}...")
         processed = preprocess_text(text, lemmatizer, stop_words, sym_spell=sym_spell_global, use_symspell=use_symspell)
         processed_texts.append(processed)
    df_copy['processed_text'] = processed_texts
    original_len = len(df_copy); df_copy = df_copy[df_copy['processed_text'].str.strip().astype(bool)]
    if len(df_copy) < original_len: logger.warning(f"Dropped {original_len - len(df_copy)} rows due to empty processed text.")
    logger.info(f"Text preprocessing complete. Shape: {df_copy.shape}")
    return df_copy


# === Fear Scoring using MacBERTh (Unchanged Class, applied to Old Bailey) ===
class MacBERThFearScorer:
    _instance = None; _model_name = MACBERTH_MODEL_NAME; _fear_words = FEAR_WORDS
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: logger.info(f"Creating MacBERThFearScorer instance..."); cls._instance = super().__new__(cls); cls._instance._initialized = False
        return cls._instance
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if self._initialized: return
        logger.info(f"Initializing MacBERTh model for embedding: {self._model_name}...")
        self.device = device if device else DEVICE; logger.info(f"MacBERTh on device: {self.device}")
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
        except Exception as e: logger.error(f"Failed init MacBERTh: {e}", exc_info=True); raise
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]; input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1); sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    @torch.no_grad()
    def calculate_fear_scores(self, texts: List[str], batch_size: int = 32) -> List[float]:
        if not self._initialized: raise RuntimeError("Scorer not initialized.")
        if not texts: return []
        all_fear_scores = []; num_texts = len(texts); logger.info(f"Calculating fear scores for {num_texts} texts...")
        log_interval = max(1, (num_texts // batch_size) // 10)
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i : i + batch_size]; valid_batch_texts = [str(t) if t else "" for t in batch_texts]
            inputs = self.tokenizer(valid_batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
            similarities = cosine_similarity(embeddings, self.average_fear_vector)
            all_fear_scores.extend(similarities.flatten().tolist())
            if (i // batch_size + 1) % log_interval == 0: logger.info(f" Scored {min(i + batch_size, num_texts)}/{num_texts}...")
        if len(all_fear_scores) != num_texts: logger.error(f"Score length mismatch: {len(all_fear_scores)} vs {num_texts}. Pad."); all_fear_scores.extend([0.0] * (num_texts - len(all_fear_scores)))
        logger.info("Fear score calculation complete.")
        return all_fear_scores

@memory.cache
def calculate_fear_scores_dataframe(df: pd.DataFrame, text_col: str = "processed_text", batch_size: int = 32) -> pd.DataFrame:
    logger.info(f"Calculating MacBERTh scores for '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Col '{text_col}' NF.")
    if df[text_col].isnull().any(): logger.warning(f"'{text_col}' has NaNs. Fill empty."); df[text_col] = df[text_col].fillna('')
    df_copy = df.copy(); scorer = MacBERThFearScorer(); texts_to_score = df_copy[text_col].tolist()
    fear_scores = scorer.calculate_fear_scores(texts_to_score, batch_size=batch_size)
    df_copy["fear_score"] = fear_scores
    logger.info("Fear scoring done."); logger.info(f"Stats: Min={np.min(fear_scores):.3f}, Max={np.max(fear_scores):.3f}, Mean={np.mean(fear_scores):.3f}, Std={np.std(fear_scores):.3f}")
    return df_copy

# === Data Aggregation and Merging (WEEKLY - UPDATED) ===
@memory.cache
def aggregate_weekly_sentiment_and_merge(text_df: pd.DataFrame, weekly_mortality_df: pd.DataFrame, agg_method: str = AGGREGATION_METHOD, fear_thresh: float = FEAR_THRESHOLD) -> pd.DataFrame:
    """
    (Cached) Aggregates WEEKLY fear scores from text data based on 'week_date',
    then merges this weekly score onto the WEEKLY mortality data. Creates lagged features.
    Standardizes the chosen feature and its lag.
    Returns a weekly DataFrame ready for TFT.
    """
    logger.info(f"Aggregating WEEKLY sentiment using '{agg_method}' and merging with WEEKLY mortality...")
    if 'fear_score' not in text_df.columns or 'week_date' not in text_df.columns: raise ValueError("'fear_score','week_date' needed in text_df.")
    if not all(c in weekly_mortality_df.columns for c in ['week_date', 'year', 'week_of_year', 'deaths']): raise ValueError("mortality_df missing cols.")

    text_df_copy = text_df[['week_date', 'fear_score']].copy()
    text_df_copy['fear_score'] = pd.to_numeric(text_df_copy['fear_score'], errors='coerce')
    text_df_copy.dropna(subset=['fear_score', 'week_date'], inplace=True)

    # --- Aggregate Fear Score by WEEK using chosen method ---
    weekly_sentiment = pd.DataFrame()
    feature_col_name = f'fear_score_weekly_{agg_method}'

    if agg_method == 'mean': weekly_sentiment = text_df_copy.groupby('week_date')['fear_score'].mean().reset_index()
    elif agg_method == 'max': weekly_sentiment = text_df_copy.groupby('week_date')['fear_score'].max().reset_index()
    elif agg_method == 'proportion':
        logger.info(f"Calculating proportion > {fear_thresh}"); text_df_copy['high_fear'] = (text_df_copy['fear_score'] > fear_thresh).astype(int)
        weekly_sentiment = text_df_copy.groupby('week_date')['high_fear'].mean().reset_index()
    else: raise ValueError(f"Unsupported agg method: {agg_method}.")
    weekly_sentiment.rename(columns={weekly_sentiment.columns[1]: feature_col_name}, inplace=True)
    logger.info(f"Aggregated weekly sentiment ({agg_method}) for {weekly_sentiment.shape[0]} weeks.")

    # --- Merge Weekly Sentiment with Weekly Mortality ---
    logger.info(f"Merging weekly '{feature_col_name}' with weekly mortality data...")
    # Ensure 'week_date' is datetime for merging
    weekly_sentiment['week_date'] = pd.to_datetime(weekly_sentiment['week_date'])
    weekly_mortality_df['week_date'] = pd.to_datetime(weekly_mortality_df['week_date'])
    # Perform an OUTER merge to keep all weeks from both datasets
    merged_df = pd.merge(weekly_mortality_df, weekly_sentiment, on='week_date', how='outer')
    # Sort by date after merge
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)

    # --- Handle Missing Data from Merge ---
    missing_sentiment = merged_df[feature_col_name].isnull().sum()
    missing_deaths = merged_df['deaths'].isnull().sum()
    if missing_sentiment > 0: logger.warning(f"{missing_sentiment} weeks have missing sentiment data. Imputing with rolling median.")
    if missing_deaths > 0: logger.warning(f"{missing_deaths} weeks have missing mortality data. Imputing with 0 (or consider rolling median).")

    # Impute missing values - use rolling window or interpolation for time series
    # Rolling median imputation for sentiment
    merged_df[feature_col_name] = merged_df[feature_col_name].fillna(merged_df[feature_col_name].rolling(4, min_periods=1, center=True).median())
    merged_df[feature_col_name].fillna(merged_df[feature_col_name].median(), inplace=True) # Fill remaining with global median

    # Impute missing deaths (e.g., with 0 or rolling median) - using 0 assumes no report means no deaths (could be wrong)
    # merged_df['deaths'].fillna(merged_df['deaths'].rolling(4, min_periods=1, center=True).median(), inplace=True)
    merged_df['deaths'].fillna(0, inplace=True) # Simpler imputation
    # Fill missing year/week after outer merge
    merged_df['year'] = merged_df['year'].fillna(merged_df['week_date'].dt.isocalendar().year)
    merged_df['week_of_year'] = merged_df['week_of_year'].fillna(merged_df['week_date'].dt.isocalendar().week)

    # --- Create Lagged Weekly Fear Score ---
    lag_weeks = 4 # Example: use 4-week lag
    feature_lag_col_name = f'{feature_col_name}_lag{lag_weeks}w'
    logger.info(f"Creating lagged feature: '{feature_lag_col_name}' ({lag_weeks} weeks)...")
    merged_df[feature_lag_col_name] = merged_df[feature_col_name].shift(lag_weeks)
    initial_nan_count = merged_df[feature_lag_col_name].isnull().sum()
    if initial_nan_count > 0:
        logger.info(f"Imputing {initial_nan_count} initial NaNs in lagged fear score (rolling median).")
        merged_df[feature_lag_col_name] = merged_df[feature_lag_col_name].fillna(merged_df[feature_lag_col_name].rolling(4, min_periods=1, center=True).median())
        merged_df[feature_lag_col_name].fillna(merged_df[feature_lag_col_name].median(), inplace=True) # Fill remaining

    # --- Standardize the Chosen Fear Scores ---
    feature_std_col_name = 'feature_std'; feature_lag_std_col_name = 'feature_lag_std'
    logger.info(f"Standardizing '{feature_col_name}' -> '{feature_std_col_name}' and '{feature_lag_col_name}' -> '{feature_lag_std_col_name}'.")
    scaler_current = StandardScaler(); merged_df[feature_std_col_name] = scaler_current.fit_transform(merged_df[[feature_col_name]])
    scaler_lagged = StandardScaler(); merged_df[feature_lag_std_col_name] = scaler_lagged.fit_transform(merged_df[[feature_lag_col_name]])

    # --- Prepare for TFT (Weekly Time Index) ---
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)
    # Time index based on weeks since start
    merged_df["time_idx"] = (merged_df["week_date"] - merged_df["week_date"].min()).dt.days // 7
    merged_df["deaths"] = merged_df["deaths"].astype(float)
    merged_df[feature_std_col_name] = merged_df[feature_std_col_name].astype(float)
    merged_df[feature_lag_std_col_name] = merged_df[feature_lag_std_col_name].astype(float)
    merged_df["year"] = merged_df["year"].astype(str)
    merged_df["week_of_year"] = merged_df["week_of_year"].astype(str)
    merged_df["series_id"] = "London"

    final_cols = [
        "week_date", "time_idx", "deaths", "year", "week_of_year", "series_id",
        feature_col_name, feature_lag_col_name, # Raw scores
        feature_std_col_name, feature_lag_std_col_name # Standardized scores
    ]
    merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]] # Ensure columns exist

    logger.info(f"Final weekly data shape for TFT: {merged_df.shape}. Time idx range: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Columns: {merged_df.columns.tolist()}")
    nan_check = merged_df.isnull().sum()
    logger.info(f"NaN check:\n{nan_check[nan_check > 0]}") # Show only columns with NaNs
    # Drop any remaining rows with NaNs in critical columns (should be few after imputation)
    critical_cols = ['time_idx', 'deaths', feature_std_col_name, feature_lag_std_col_name]
    merged_df.dropna(subset=critical_cols, inplace=True)
    logger.info(f"Shape after final NaN drop: {merged_df.shape}")

    if merged_df["time_idx"].max() < WEEKLY_MAX_ENCODER_LENGTH + WEEKLY_MAX_PREDICTION_LENGTH:
         raise ValueError(f"Insufficient weekly data span ({merged_df.shape[0]} weeks) for TFT config (enc={WEEKLY_MAX_ENCODER_LENGTH}, pred={WEEKLY_MAX_PREDICTION_LENGTH}).")
    return merged_df


# -----------------------------------------------------------------------------
# 5. TFT Training and Evaluation (WEEKLY)
# -----------------------------------------------------------------------------
def train_tft_model(df: pd.DataFrame,
                    max_epochs: int = MAX_EPOCHS,
                    batch_size: int = BATCH_SIZE,
                    encoder_length: int = WEEKLY_MAX_ENCODER_LENGTH, # Use weekly param
                    pred_length: int = WEEKLY_MAX_PREDICTION_LENGTH, # Use weekly param
                    lr: float = LEARNING_RATE,
                    hidden_size: int = HIDDEN_SIZE,
                    attn_heads: int = ATTENTION_HEAD_SIZE,
                    dropout: float = DROPOUT,
                    hidden_cont_size: int = HIDDEN_CONTINUOUS_SIZE,
                    clip_val: float = GRADIENT_CLIP_VAL) -> Tuple[Optional[TemporalFusionTransformer], Optional[pl.Trainer], Optional[torch.utils.data.DataLoader], Optional[TimeSeriesDataSet]]:
    """Trains the Temporal Fusion Transformer model on WEEKLY data using default scaling."""

    logger.info(f"Setting up WEEKLY TFT model training...")
    logger.info(f" Encoder length: {encoder_length} weeks, Prediction length: {pred_length} weeks")

    max_idx = df["time_idx"].max(); training_cutoff = max_idx - pred_length
    logger.info(f"Weekly Data Cutoff for Training: time_idx <= {training_cutoff}")
    if training_cutoff < encoder_length: logger.error(f"Cutoff {training_cutoff} < Encoder length {encoder_length}. Need more data."); return None, None, None, None

    logger.info("Ensuring correct dtypes before TimeSeriesDataSet...")
    try:
        data_for_tft = df.copy()
        numeric_cols = ["time_idx", "deaths", "feature_std", "feature_lag_std"] # Use generic names
        for col in numeric_cols: data_for_tft[col] = pd.to_numeric(data_for_tft[col], errors='coerce')
        categorical_cols = ["series_id", "week_of_year", "year"] # Use week_of_year
        for col in categorical_cols: data_for_tft[col] = data_for_tft[col].astype(str)
        if data_for_tft[numeric_cols].isnull().any().any():
             logger.warning(f"NaNs found after casting: \n{data_for_tft[numeric_cols].isnull().sum()}. Imputing with median.")
             for col in numeric_cols: data_for_tft[col].fillna(data_for_tft[col].median(), inplace=True)
        logger.info("Dtype check passed.")
    except Exception as e: logger.error(f"Error during dtype check: {e}", exc_info=True); return None, None, None, None

    logger.info("Setting up WEEKLY TimeSeriesDataSet for TFT (Using Default Normalization)...")
    try:
        training_dataset = TimeSeriesDataSet(
            data_for_tft[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx", target="deaths", group_ids=["series_id"],
            max_encoder_length=encoder_length, max_prediction_length=pred_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["week_of_year"], # Week of year is known
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=["year"], # Year can be unknown
            time_varying_unknown_reals=[ "deaths", "feature_std", "feature_lag_std" ], # Use generic names
            add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=True,
            # Handle potential NaNs in categoricals robustly
            categorical_encoders={"year": NaNLabelEncoder(add_nan=True), "week_of_year": NaNLabelEncoder(add_nan=True)}
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
            dropout=dropout, hidden_continuous_size=hidden_cont_size, loss=loss_metric, log_interval=50, # Log less often for weekly
            optimizer="adam", reduce_on_plateau_patience=5,
        )
        logger.info(f"TFT model parameters: {tft.size()/1e6:.1f} million")
    except Exception as e: logger.error(f"Error initializing TFT: {e}", exc_info=True); return None, None, val_dataloader, validation_dataset

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accelerator, devices = ('cpu', 1) # Hardcoded CPU
    logger.info(f"Configuring Trainer (Accelerator: {accelerator}, Devices: {devices})...")
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name="tft_fear_model_weekly") # New logger name
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, gradient_clip_val=clip_val, callbacks=[lr_monitor, early_stop_callback], logger=tb_logger, enable_progress_bar=True)

    logger.info("Starting TFT model training (Weekly)...")
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

# --- evaluate_model function (UPDATED FOR WEEKLY) ---
def evaluate_model(model: TemporalFusionTransformer, dataloader: torch.utils.data.DataLoader, dataset: TimeSeriesDataSet, plot_dir: str) -> Dict[str, float]:
    """Evaluates TFT model, returns metrics, saves plots for WEEKLY aggregated forecast."""
    logger.info("Evaluating model performance on validation set (Weekly)...")
    # ... (Metric calculation logic remains the same as previous version) ...
    results = {}
    if model is None or dataloader is None or len(dataloader) == 0: logger.error("Model/Dataloader missing for eval."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
    try: eval_device = next(model.parameters()).device
    except Exception: eval_device = torch.device(DEVICE); model.to(eval_device)
    logger.info(f"Evaluation device: {eval_device}")
    actuals_metric_list, preds_metric_list = [], []
    with torch.no_grad():
        for x, y in iter(dataloader):
            x = {k: v.to(eval_device) for k, v in x.items() if isinstance(v, torch.Tensor)}; target = y[0].to(eval_device)
            preds = model(x)["prediction"]; preds_metric_list.append(preds[:, :, 1].cpu()); actuals_metric_list.append(target.cpu())
    if not preds_metric_list: logger.error("No predictions collected."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
    actuals_flat_m = torch.cat(actuals_metric_list).flatten().numpy(); preds_median_flat_m = torch.cat(preds_metric_list).flatten().numpy()
    min_len_m = min(len(actuals_flat_m), len(preds_median_flat_m))
    if len(actuals_flat_m) != len(preds_median_flat_m): logger.warning(f"Metric length mismatch: Truncate."); preds_median_flat_m=preds_median_flat_m[:min_len_m]; actuals_flat_m=actuals_flat_m[:min_len_m]
    preds_median_flat_m = np.maximum(0, preds_median_flat_m)
    val_mae = mean_absolute_error(actuals_flat_m, preds_median_flat_m); val_mse = mean_squared_error(actuals_flat_m, preds_median_flat_m)
    denominator = (np.abs(actuals_flat_m) + np.abs(preds_median_flat_m)) / 2.0; val_smape = np.mean(np.abs(preds_median_flat_m - actuals_flat_m) / np.where(denominator == 0, 1, denominator)) * 100
    results = {"MAE": val_mae, "MSE": val_mse, "SMAPE": val_smape}
    logger.info(f"[Validation Metrics] MAE={val_mae:.3f} MSE={val_mse:.3f} SMAPE={val_smape:.3f}%")

    # --- Plotting Section (Using predict for aligned data) ---
    logger.info("Generating weekly evaluation plots...")
    try:
        prediction_output = model.predict(dataloader, mode="raw", return_x=True, return_index=True)
        preds_dict = None; x_dict = None; index_df = None
        if isinstance(prediction_output, (list, tuple)) and len(prediction_output) >= 1:
            if isinstance(prediction_output[0], dict): preds_dict = prediction_output[0]
            if len(prediction_output) > 1 and isinstance(prediction_output[1], dict): x_dict = prediction_output[1]
            if len(prediction_output) > 2 and isinstance(prediction_output[2], pd.DataFrame): index_df = prediction_output[2]
        elif isinstance(prediction_output, dict): preds_dict = prediction_output
        if preds_dict is None or x_dict is None or index_df is None: logger.error(f"Predict unpack fail. Skip plots."); return results

        preds_tensor = preds_dict['prediction'].cpu(); p10_flat = preds_tensor[:, :, 0].flatten().numpy(); p50_flat = preds_tensor[:, :, 1].flatten().numpy(); p90_flat = preds_tensor[:, :, 2].flatten().numpy()
        actuals_flat = x_dict['decoder_target'].flatten().cpu().numpy(); time_idx_flat = index_df['time_idx'].values
        n_preds = len(p50_flat); n_actuals = len(actuals_flat); n_time = len(time_idx_flat)
        logger.debug(f"Plot shapes - Preds: {n_preds}, Actuals: {n_actuals}, Time: {n_time}")
        if not (n_preds == n_actuals == n_time):
            logger.warning(f"Plot length mismatch! Truncate."); min_len_plot = min(n_preds, n_actuals, n_time)
            if min_len_plot == 0: logger.error("Zero length plot data. Skip plots."); return results
            p10_flat=p10_flat[:min_len_plot]; p50_flat=p50_flat[:min_len_plot]; p90_flat=p90_flat[:min_len_plot]; actuals_flat=actuals_flat[:min_len_plot]; time_idx_flat=time_idx_flat[:min_len_plot]
        if len(np.unique(time_idx_flat)) <= 1: logger.warning("Plot data has <= 1 unique time index.")
        p10_flat = np.maximum(0, p10_flat); p50_flat = np.maximum(0, p50_flat); p90_flat = np.maximum(0, p90_flat)
        sort_indices = np.argsort(time_idx_flat); time_idx_sorted=time_idx_flat[sort_indices]; actuals_sorted=actuals_flat[sort_indices]; p10_sorted=p10_flat[sort_indices]; p50_sorted=p50_flat[sort_indices]; p90_sorted=p90_flat[sort_indices]

        # --- Generate Plots ---
        try: start_date_dt = dataset.data["raw"]["week_date"].min(); x_label = f"Time Index (Weeks since {start_date_dt:%Y-%m-%d})"
        except Exception: x_label = "Time Index (Weeks)"

        plt.figure(figsize=(15, 7)); plt.plot(time_idx_sorted, actuals_sorted, label="Actual Deaths", marker='.', linestyle='-', alpha=0.7, color='black', markersize=3, linewidth=0.8) # Thinner lines for weekly
        plt.plot(time_idx_sorted, p50_sorted, label="Predicted Median (p50)", linestyle='--', alpha=0.9, color='tab:orange', linewidth=1.2)
        plt.fill_between(time_idx_sorted, p10_sorted, p90_sorted, color='tab:orange', alpha=0.3, label='p10-p90 Quantiles')
        plt.title(f"TFT Aggregated Forecast vs Actuals (Validation Set - Weekly)\nMAE={val_mae:.2f}, SMAPE={val_smape:.2f}%", fontsize=14)
        plt.xlabel(x_label, fontsize=12); plt.ylabel("Weekly Deaths", fontsize=12) # Updated label
        plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        plot_file = os.path.join(plot_dir, "tft_val_forecast_aggregated_weekly.png"); plt.savefig(plot_file); logger.info(f"Saved aggregated forecast plot to {plot_file}"); plt.close()

        residuals = actuals_sorted - p50_sorted
        plt.figure(figsize=(10, 6)); plt.scatter(p50_sorted, residuals, alpha=0.3, s=15, color='tab:blue', edgecolors='k', linewidth=0.5)
        plt.axhline(0, color='red', linestyle='--', linewidth=1); plt.title('Residual Plot (Actuals - Median Predictions)', fontsize=14)
        plt.xlabel('Predicted Median Deaths', fontsize=12); plt.ylabel('Residuals', fontsize=12); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        save_path_res = os.path.join(plot_dir, "residuals_weekly_plot.png"); plt.savefig(save_path_res); logger.info(f"Saved residual plot to {save_path_res}"); plt.close()
    except Exception as e: logger.warning(f"Evaluation plotting failed: {e}", exc_info=True)
    finally: plt.close('all')
    return results

# -----------------------------------------------------------------------------
# 6. Enhanced Plotting Functions (UPDATED FOR WEEKLY)
# -----------------------------------------------------------------------------
def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str, ylabel: str, filename: str, plot_dir: str):
    """Plots a simple time series using a specified time column (weekly focus)."""
    logger.info(f"Generating plot: {title}")
    if time_col not in df.columns or value_col not in df.columns: logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        plt.figure(figsize=(18, 6)) # Wider plot for weekly data
        plt.plot(df[time_col], df[value_col], marker='.', linestyle='-', markersize=1.5, alpha=0.6, linewidth=0.8) # Smaller markers, thinner line
        plt.title(title, fontsize=14); plt.xlabel(time_col.replace('_', ' ').title(), fontsize=12); plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

def plot_dual_axis(df: pd.DataFrame, time_col: str, col1: str, col2: str, label1: str, label2: str, title: str, filename: str, plot_dir: str):
    """Plots two time series on a dual-axis chart (weekly focus)."""
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [time_col, col1, col2]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    fig, ax1 = plt.subplots(figsize=(18, 6)) # Wider plot
    try:
        time_data = df[time_col]; x_label = time_col.replace('_', ' ').title()
        if pd.api.types.is_datetime64_any_dtype(time_data): pass
        else: start_date_str = df['week_date'].min().strftime('%Y-%m-%d'); x_label = f"Time Index (Weeks since {start_date_str})"
        color1 = 'tab:blue'; ax1.set_xlabel(x_label, fontsize=12); ax1.set_ylabel(label1, color=color1, fontsize=12)
        line1 = ax1.plot(time_data, df[col1], color=color1, label=label1, alpha=0.8, linewidth=1.2)
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax2 = ax1.twinx(); color2 = 'tab:red'; ax2.set_ylabel(label2, color=color2, fontsize=12)
        line2 = ax2.plot(time_data, df[col2], color=color2, label=label2, linestyle='--', alpha=0.8, linewidth=1.2)
        ax2.tick_params(axis='y', labelcolor=color2); fig.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.title(title, fontsize=14);
        lines = line1 + line2; labels = [l.get_label() for l in lines]; ax1.legend(lines, labels, loc='upper left')
        plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close(fig)

def plot_scatter_fear_vs_deaths(df: pd.DataFrame, fear_col: str, death_col: str, title: str, filename: str, plot_dir: str):
    """Plots a scatter plot of fear score vs deaths (weekly focus)."""
    logger.info(f"Generating plot: {title}")
    if not all(c in df.columns for c in [fear_col, death_col]): logger.error(f"Plot fail '{filename}': Missing cols."); return
    try:
        plt.figure(figsize=(8, 8)); sns.scatterplot(data=df, x=fear_col, y=death_col, alpha=0.2, s=10, edgecolor=None) # More transparency for weekly
        corr = df[fear_col].corr(df[death_col]); plt.title(f"{title}\n(Correlation: {corr:.2f})")
        plt.xlabel(fear_col.replace('_', ' ').title()); plt.ylabel(death_col.replace('_', ' ').title())
        plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Plot fail {filename}: {e}", exc_info=True)
    finally: plt.close()

# Removed plot_monthly_boxplot as it's less relevant for weekly data compared to week_of_year plots if needed

def plot_weekly_boxplot(df: pd.DataFrame, column: str, title: str, filename: str, plot_dir: str):
    """Plots a boxplot grouped by week of year."""
    logger.info(f"Generating plot: {title}")
    if 'week_of_year' not in df.columns or column not in df.columns: logger.error(f"Plot fail '{filename}': Missing 'week_of_year' or '{column}'."); return
    try:
        df['week_num'] = df['week_of_year'].astype(int)
        week_order = [str(i) for i in range(1, 54)] # Weeks 1 to 53
        plt.figure(figsize=(18, 7)) # Wider plot
        sns.boxplot(x='week_of_year', y=column, data=df, order=week_order, showfliers=False, palette="coolwarm")
        plt.title(title, fontsize=14); plt.xlabel("Week of Year", fontsize=12); plt.ylabel(column.replace('_', ' ').title(), fontsize=12)
        # Reduce number of x-axis labels shown
        tick_freq = 5
        plt.xticks(ticks=range(0, 53, tick_freq), labels=[str(i) for i in range(1, 54, tick_freq)], rotation=45, ha='right')
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, filename)); logger.info(f"Saved plot: {filename}")
    except Exception as e: logger.error(f"Failed weekly boxplot {filename}: {e}", exc_info=True)
    finally: plt.close()


# -----------------------------------------------------------------------------
# 7. Interpretation & Granger Causality (Unchanged Functions)
# -----------------------------------------------------------------------------
def interpret_tft(model: TemporalFusionTransformer, val_dataloader: torch.utils.data.DataLoader, plot_dir: str):
    """Calculates and saves TFT feature importance plots."""
    logger.info("Calculating TFT feature importance...")
    if model is None or val_dataloader is None or len(val_dataloader) == 0: logger.warning("Model/Dataloader missing, skip interpretation."); return
    try:
        interpret_device = "cpu"; model.to(interpret_device); logger.info(f"Running interpretation on: {interpret_device}")
        prediction_output = model.predict(val_dataloader, mode="raw", return_x=True)
        if isinstance(prediction_output, (list, tuple)) and len(prediction_output) >= 1 and isinstance(prediction_output[0], dict):
            raw_predictions_dict = {k: v.to(interpret_device) if isinstance(v, torch.Tensor) else v for k, v in prediction_output[0].items()}
        elif isinstance(prediction_output, dict): raw_predictions_dict = {k: v.to(interpret_device) if isinstance(v, torch.Tensor) else v for k, v in prediction_output.items()}
        else: logger.error(f"Unexpected predict() output type: {type(prediction_output)}. Skip interpretation."); return
        interpretation = model.interpret_output(raw_predictions_dict, reduction="mean")
        logger.info("Plotting TFT interpretation...")
        fig_imp = model.plot_interpretation(interpretation, plot_type="variable_importance"); fig_imp.suptitle("TFT Feature Importance (Weekly)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_path_imp = os.path.join(plot_dir, "tft_interpretation_importance_weekly.png"); fig_imp.savefig(save_path_imp); logger.info(f"Saved interpretation plot to {save_path_imp}"); plt.close(fig_imp)
    except ImportError: logger.warning("matplotlib issue. Skip interpretation plot.")
    except Exception as e: logger.error(f"Error during interpretation: {e}", exc_info=True)
    finally: model.to(DEVICE); plt.close('all')

def run_granger_causality(df: pd.DataFrame, var1: str, var2: str, max_lag: int = 12) -> Optional[Dict[int, float]]:
    """Performs Granger Causality tests using first differences. Returns {lag: p_value}."""
    logger.info(f"Running Granger Causality test: '{var1}' -> '{var2}'? (Max lag: {max_lag} weeks)") # Updated label
    if var1 not in df.columns or var2 not in df.columns: logger.error(f"Granger cols missing. Skip."); return None
    data = df[[var1, var2]].copy(); data.dropna(inplace=True) # Drop rows with any NaNs in these cols
    if data.shape[0] < max_lag + 5: logger.error(f"Not enough data ({data.shape[0]}) for Granger test after dropna. Skip."); return None
    try: data_diff = data.diff().dropna()
    except Exception as e: logger.error(f"Granger differencing error: {e}. Skip."); return None
    if data_diff.shape[0] < max_lag + 5: logger.error(f"Not enough data ({data_diff.shape[0]}) after differencing. Skip."); return None
    logger.info("Using first differences for Granger test.")
    try:
        gc_result = grangercausalitytests(data_diff[[var2, var1]], maxlag=max_lag, verbose=False)
        p_values = {lag: gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}
        significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
        if significant_lags: logger.info(f" Granger Result ('{var1}' -> '{var2}'): Significant at lags: {significant_lags} (p < 0.05).")
        else: logger.info(f" Granger Result ('{var1}' -> '{var2}'): Not significant up to lag {max_lag} (p >= 0.05).")
        return p_values
    except Exception as e: logger.error(f"Granger test error: {e}", exc_info=True); return None

def plot_granger_causality_results(p_values_dict: Dict[str, Optional[Dict[int, float]]], title_prefix: str, plot_dir: str):
    """Plots the p-values from Granger Causality tests."""
    if not p_values_dict: logger.warning("No Granger results to plot."); return
    valid_results = {k: v for k, v in p_values_dict.items() if v is not None}
    num_tests = len(valid_results)
    if num_tests == 0: logger.warning("No valid Granger results to plot."); return
    fig, axes = plt.subplots(nrows=num_tests, ncols=1, figsize=(10, 4 * num_tests), squeeze=False); axes = axes.flatten()
    fig.suptitle(f"{title_prefix} Granger Causality P-Values (Weekly Lags)", fontsize=14, y=1.02)
    for i, (test_description, p_values) in enumerate(valid_results.items()):
        ax = axes[i]; lags = list(p_values.keys()); pvals = list(p_values.values())
        bars = ax.bar(lags, pvals, color='lightblue', edgecolor='black'); ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p = 0.05 Threshold')
        for lag_idx, p in enumerate(pvals):
            if p < 0.05: bars[lag_idx].set_color('salmon'); bars[lag_idx].set_edgecolor('red')
        ax.set_title(f"Test: {test_description}"); ax.set_xlabel("Lag (Weeks)"); ax.set_ylabel("P-value"); ax.set_xticks(lags); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    filename = f"{title_prefix.lower().replace(' ','_')}_granger_causality_weekly.png"
    save_path = os.path.join(plot_dir, filename); plt.savefig(save_path); logger.info(f"Saved Granger plot to {save_path}"); plt.close(fig)

# -----------------------------------------------------------------------------
# 8. MAIN Execution Logic (UPDATED FOR OLD BAILEY & WEEKLY)
# -----------------------------------------------------------------------------
def main():
    """Main script execution flow using Old Bailey and Weekly data."""
    logger.info("--- Starting Historical Analysis Script (Old Bailey - Weekly Focus) ---")
    script_start_time = time.time()
    final_df = None; best_model = None; trainer = None; val_dl = None; val_ds = None

    try:
        os.makedirs(PLOT_DIR, exist_ok=True); logger.info(f"Plots directory: {PLOT_DIR}")

        # --- Step 1: Parse Old Bailey Data ---
        logger.info("--- Step 1: Parsing Old Bailey Sessions Papers ---")
        ob_df_raw = parse_old_bailey_papers(ob_dir=OLD_BAILEY_DIR, start_year=START_YEAR, end_year=END_YEAR)
        if not isinstance(ob_df_raw, pd.DataFrame) or ob_df_raw.empty: raise ValueError("Step 1 Failed: No Old Bailey documents found/parsed.")

        # --- Step 2: Preprocess Text ---
        logger.info("--- Step 2: Preprocessing Old Bailey Text ---")
        USE_SYMSPELL_CORRECTION = False # Keep False
        ob_df_processed = preprocess_text_dataframe(ob_df_raw, text_col='text', use_symspell=USE_SYMSPELL_CORRECTION)
        if ob_df_processed.empty: raise ValueError("Step 2 Failed: Text preprocessing resulted in empty DataFrame.")
        del ob_df_raw

        # --- Step 3: Calculate Fear Scores ---
        logger.info("--- Step 3: Calculating Fear Scores using MacBERTh ---")
        # Adjust batch size if needed for potentially longer Old Bailey texts
        ob_df_scored = calculate_fear_scores_dataframe(ob_df_processed, text_col='processed_text', batch_size=BATCH_SIZE // 2)
        if 'fear_score' not in ob_df_scored.columns: raise ValueError("Step 3 Failed: Fear scoring failed.")
        del ob_df_processed

        # --- Step 4: Load Weekly Mortality ---
        logger.info("--- Step 4: Loading and Processing Weekly Mortality Data ---")
        weekly_mortality_df = load_and_aggregate_weekly_mortality(file_path=COUNTS_FILE, start_year=START_YEAR, end_year=END_YEAR)
        if weekly_mortality_df.empty: raise ValueError("Step 4 Failed: No weekly mortality data found.")

        # --- Step 5: Aggregate Sentiment (Weekly) & Merge ---
        logger.info(f"--- Step 5: Aggregating Weekly Sentiment ('{AGGREGATION_METHOD}') and Merging ---")
        final_df = aggregate_weekly_sentiment_and_merge(ob_df_scored, weekly_mortality_df, agg_method=AGGREGATION_METHOD, fear_thresh=FEAR_THRESHOLD)
        if final_df.empty: raise ValueError("Step 5 Failed: Merging resulted in empty DataFrame.")
        del ob_df_scored; del weekly_mortality_df

        # Save final weekly data
        try: final_df.to_csv(os.path.join(PLOT_DIR, f"final_weekly_data_fear_{AGGREGATION_METHOD}.csv"), index=False); logger.info("Saved final weekly data.")
        except Exception as e: logger.warning(f"Could not save final data CSV: {e}")

        # --- Step 5.5: Generate Data Plots (Weekly) ---
        logger.info("--- Step 5.5: Generating Data Exploration Plots (Weekly) ---")
        raw_fear_col = f'fear_score_weekly_{AGGREGATION_METHOD}'
        lag_weeks = 4 # Match lag used in aggregation
        lag_fear_col = f'{raw_fear_col}_lag{lag_weeks}w'
        std_fear_col = 'feature_std'
        lag_std_fear_col = 'feature_lag_std'

        if final_df is not None and not final_df.empty:
            plot_time_series(final_df, 'week_date', 'deaths', 'Weekly Deaths Over Time', 'Total Deaths', 'deaths_weekly_timeseries.png', PLOT_DIR)
            plot_time_series(final_df, 'week_date', raw_fear_col, f'Weekly Fear ({AGGREGATION_METHOD.title()})', 'Fear Score', f'fear_weekly_{AGGREGATION_METHOD}_timeseries.png', PLOT_DIR)
            plot_time_series(final_df, 'week_date', lag_fear_col, f'Lagged ({lag_weeks}w) Weekly Fear ({AGGREGATION_METHOD.title()})', 'Fear Score', f'fear_weekly_{AGGREGATION_METHOD}_lag{lag_weeks}w_timeseries.png', PLOT_DIR)
            plot_dual_axis(final_df, 'week_date', 'deaths', std_fear_col, 'Weekly Deaths', f'Std. Weekly Fear ({AGGREGATION_METHOD.title()})', f'Weekly Deaths vs Std. Weekly Fear ({AGGREGATION_METHOD})', f'deaths_vs_fear_{AGGREGATION_METHOD}_std_weekly_dual_axis.png', PLOT_DIR)
            plot_scatter_fear_vs_deaths(final_df, lag_fear_col, 'deaths', f'Weekly Deaths vs Lagged ({lag_weeks}w) Fear ({AGGREGATION_METHOD.title()})', f'scatter_deaths_vs_fear_{AGGREGATION_METHOD}_lag{lag_weeks}w.png', PLOT_DIR)
            plot_weekly_boxplot(final_df, 'deaths', 'Weekly Distribution of Deaths (by Week of Year)', 'boxplot_deaths_weekly.png', PLOT_DIR)
        else: logger.warning("final_df empty, skipping data plots.")

        # === Step 6: Train and Evaluate TFT Model (Weekly) ===
        logger.info("--- Step 6: Training and Evaluating TFT Model (Weekly) ---")
        if final_df is not None and not final_df.empty:
            best_model, trainer, val_dl, val_ds = train_tft_model(
                df=final_df,
                encoder_length=WEEKLY_MAX_ENCODER_LENGTH, # Pass weekly params
                pred_length=WEEKLY_MAX_PREDICTION_LENGTH
            )
            if best_model and val_dl and val_ds:
                logger.info("--- Step 6.5: Evaluating Best TFT Model ---")
                eval_metrics = evaluate_model(best_model, val_dl, val_ds, plot_dir=PLOT_DIR)
                logger.info(f"Final Validation Metrics: {eval_metrics}")
            else: logger.error("TFT Model training failed or did not return model/dataloader.")
        else: logger.error("Final DataFrame empty, cannot train TFT model.")

        # === Step 7: Interpretation and Granger Causality (Weekly) ===
        logger.info("--- Step 7: Model Interpretation & Granger Causality (Weekly) ---")
        if final_df is not None and not final_df.empty:
            if best_model and val_dl: interpret_tft(best_model, val_dl, PLOT_DIR)
            else: logger.warning("Skipping TFT interpretation.")

            logger.info("--- Running Granger Causality Tests (Weekly) ---")
            granger_results = {}
            # Test with appropriate weekly lags (e.g., up to 8 weeks)
            max_weekly_lag = 8
            desc1 = f"'{raw_fear_col}' -> 'deaths'"
            granger_results[desc1] = run_granger_causality(final_df, raw_fear_col, 'deaths', max_lag=max_weekly_lag)
            desc2 = f"'{lag_fear_col}' -> 'deaths'" # Using the specific lagged feature
            granger_results[desc2] = run_granger_causality(final_df, lag_fear_col, 'deaths', max_lag=max_weekly_lag)
            desc3 = f"'deaths' -> '{raw_fear_col}'"
            granger_results[desc3] = run_granger_causality(final_df, 'deaths', raw_fear_col, max_lag=max_weekly_lag)

            plot_granger_causality_results(granger_results, title_prefix=f"Weekly Fear ({AGGREGATION_METHOD}) vs Deaths", plot_dir=PLOT_DIR)
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