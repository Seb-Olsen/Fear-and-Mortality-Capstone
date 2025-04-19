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
AGGREGATION_METHOD = 'max'
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

@memory.cache
def parse_old_bailey_papers(ob_dir: str = OLD_BAILEY_DIR, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    (Cached) Parses Old Bailey Sessions Papers XML files (TEI.2 format).
    Extracts session start date, text, AND structured trial data
    (primary offence category, verdict category, punishment category) for each trial account.
    Maps date to the start of the ISO week (Monday).
    Returns DataFrame with ['week_date', 'doc_id', 'trial_id', 'text', 'offence_cat', 'verdict_cat', 'punishment_cat'].
    """
    records = []
    logger.info(f"Starting Old Bailey Sessions Papers parsing (Years: {start_year}-{end_year}) - Extracting Structured Data...")
    file_count = 0; processed_trials = 0; skipped_date = 0; parse_errors = 0; date_parse_attempts = 0

    for rootdir, _, files in os.walk(ob_dir):
        for fname in files:
            if not (fname.endswith('.xml') or re.match(r'^\d{8}$', fname) or '.' not in fname): continue
            file_count += 1
            if file_count % 500 == 0: logger.info(f" Scanning file {file_count}...")

            fpath = os.path.join(rootdir, fname)
            doc_id_base = os.path.splitext(fname)[0] # Use filename as base doc id

            try:
                tree = ET.parse(fpath)
                root_tei = tree.getroot() # <TEI.2>

                # --- Extract Session Date (from <div0 type="sessionsPaper">) ---
                session_div = root_tei.find('.//div0[@type="sessionsPaper"]')
                session_date_str = None
                session_date = None
                doc_id = doc_id_base # Default doc_id

                if session_div is not None:
                    # Use ID from session_div if available and looks like a date
                    if 'id' in session_div.attrib and re.match(r'^\d{8}$', session_div.attrib['id']):
                        session_date_str = session_div.attrib['id']
                        doc_id = session_date_str # Prefer date string as doc_id
                        date_parse_attempts += 1
                    # Fallback to interp date if div0 id is missing/invalid
                    interp_date_node = session_div.find('.//interp[@type="date"]')
                    if session_date_str is None and interp_date_node is not None and 'value' in interp_date_node.attrib:
                         session_date_str = interp_date_node.attrib['value']
                         if re.match(r'^\d{8}$', session_date_str):
                            doc_id = session_date_str # Use date string as doc_id
                            date_parse_attempts += 1
                         else: session_date_str = None # Ignore invalid date format

                if session_date_str:
                    try:
                        session_date = datetime.strptime(session_date_str, '%Y%m%d')
                        if not (start_year <= session_date.year <= end_year):
                            skipped_date += 1; continue
                    except ValueError:
                        logger.warning(f"Date parse failed '{session_date_str}' in {fname}. Skip file."); skipped_date += 1; continue
                else:
                    logger.debug(f"No valid session date found in {fname}. Skip file."); skipped_date += 1; continue

                # Map Session Date to Start of ISO Week (Monday)
                iso_year, iso_week, _ = session_date.isocalendar()
                try: week_start_date = datetime.fromisocalendar(iso_year, iso_week, 1)
                except ValueError: logger.warning(f"Week start date fail for {session_date_str} in {fname}. Skip file."); skipped_date += 1; continue

                # Check against pandas min date
                min_pandas_date = datetime(1678, 1, 1)
                if week_start_date < min_pandas_date:
                    # logger.debug(f"Skipping file {fname}, date {week_start_date} before pandas min.")
                    skipped_date += 1; continue

                # --- Iterate through Trial Accounts (<div1 type="trialAccount">) ---
                trial_accounts = root_tei.findall('.//div1[@type="trialAccount"]')
                if not trial_accounts:
                    # logger.debug(f"No trial accounts found in {fname}.")
                    continue

                for trial_div in trial_accounts:
                    trial_id = trial_div.get('id', f"{doc_id}_trial_{processed_trials+1}") # Get trial ID or create one

                    # Extract Text for this trial
                    trial_text_parts = []
                    for p_node in trial_div.findall('.//p'):
                        node_text = ' '.join(t.strip() for t in p_node.itertext() if t and t.strip())
                        node_text_clean = re.sub(r'\s+', ' ', node_text).strip()
                        if node_text_clean: trial_text_parts.append(node_text_clean)
                    trial_text = ' '.join(trial_text_parts) if trial_text_parts else ""

                    # Extract Structured Info (taking the first one found for simplicity, might need refinement)
                    offence_cat = None; verdict_cat = None; punishment_cat = None

                    offence_interp = trial_div.find('.//rs[@type="offenceDescription"]/interp[@type="offenceCategory"]')
                    if offence_interp is not None and 'value' in offence_interp.attrib:
                        offence_cat = offence_interp.get('value')

                    verdict_interp = trial_div.find('.//rs[@type="verdictDescription"]/interp[@type="verdictCategory"]')
                    if verdict_interp is not None and 'value' in verdict_interp.attrib:
                        verdict_cat = verdict_interp.get('value')

                    punishment_interp = trial_div.find('.//rs[@type="punishmentDescription"]/interp[@type="punishmentCategory"]')
                    if punishment_interp is not None and 'value' in punishment_interp.attrib:
                        punishment_cat = punishment_interp.get('value')

                    # Only add record if we have at least an offence or verdict
                    if offence_cat or verdict_cat:
                        records.append({
                            'week_date': week_start_date,
                            'doc_id': doc_id, # Session document ID
                            'trial_id': trial_id, # Individual trial ID
                            'text': trial_text, # Text specific to this trial (optional)
                            'offence_cat': offence_cat,
                            'verdict_cat': verdict_cat,
                            'punishment_cat': punishment_cat
                        })
                        processed_trials += 1
                        if processed_trials % 1000 == 0: logger.info(f" Found {processed_trials} valid trial records...")

            except ET.ParseError as e: logger.warning(f"XML Parse Error {fname}: {e}"); parse_errors += 1
            except Exception as e: logger.warning(f"General Error processing {fname}: {e}", exc_info=False); parse_errors += 1

    logger.info(f"Finished Old Bailey parsing. Files scanned: {file_count}")
    if processed_trials == 0: logger.error("CRITICAL: No trial records processed. Check XML structure or parsing logic.")
    else: logger.info(f" Processed {processed_trials} valid trial account records.")
    logger.info(f" Files skipped due to date issues: {skipped_date}.")
    logger.info(f" Errors during parsing: {parse_errors}.")
    if not records: return pd.DataFrame(columns=['week_date', 'doc_id', 'trial_id', 'text', 'offence_cat', 'verdict_cat', 'punishment_cat'])

    # Create DataFrame
    df = pd.DataFrame(records)
    df['week_date'] = pd.to_datetime(df['week_date']) # Convert date column
    # Clean categories slightly
    for col in ['offence_cat', 'verdict_cat', 'punishment_cat']:
        df[col] = df[col].str.lower().str.strip().fillna('unknown')

    logger.info(f"Old Bailey Structured DataFrame prepared: {df.shape[0]} trial records. Date Range: {df['week_date'].min():%Y-%m-%d} to {df['week_date'].max():%Y-%m-%d}")
    logger.info(f"Sample Offence Categories: {df['offence_cat'].value_counts().head().to_dict()}")
    logger.info(f"Sample Verdict Categories: {df['verdict_cat'].value_counts().head().to_dict()}")
    logger.info(f"Sample Punishment Categories: {df['punishment_cat'].value_counts().head().to_dict()}")

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
def aggregate_combined_metrics(trial_df_scored: pd.DataFrame, weekly_mortality_df: pd.DataFrame) -> pd.DataFrame:
    """
    (Cached) Aggregates weekly structured trial metrics AND conditional MacBERTh sentiment scores.
    Merges with WEEKLY mortality data, log-transforms deaths, creates lagged features, clips ends, standardizes.
    Returns a weekly DataFrame ready for analysis/TFT.
    """
    logger.info(f"Aggregating WEEKLY combined structured & sentiment metrics...")

    required_trial_cols = ['week_date', 'trial_id', 'offence_cat', 'verdict_cat', 'punishment_cat', 'fear_score']
    if not all(c in trial_df_scored.columns for c in required_trial_cols):
        raise ValueError(f"trial_df_scored missing required columns. Need: {required_trial_cols}. Found: {trial_df_scored.columns.tolist()}")
    if not all(c in weekly_mortality_df.columns for c in ['week_date', 'year', 'week_of_year', 'deaths']):
        raise ValueError("mortality_df missing required columns.")

    trial_df_scored['week_date'] = pd.to_datetime(trial_df_scored['week_date'])
    trial_df_scored['fear_score'] = pd.to_numeric(trial_df_scored['fear_score'], errors='coerce')
    trial_df_scored.dropna(subset=['fear_score'], inplace=True) # Drop trials where fear score couldn't be calculated

    # Define categories (same as before)
    VIOLENT_CATS = ['violenttheft', 'kill', 'sexual']
    PROPERTY_CATS = ['theft', 'deception', 'damage', 'royaloffences']
    GUILTY_VERDICTS = ['guilty']
    NOT_GUILTY_VERDICTS = ['notguilty', 'unknown']
    DEATH_PUNISH = ['death']
    TRANSPORT_PUNISH = ['transport']
    CORPORAL_PUNISH = ['corporal', 'miscpunish']

    # --- Calculate Weekly Aggregations ---
    # Use pivot_table or multiple groupbys to get conditional means

    # Group by week first
    grouped_week = trial_df_scored.groupby('week_date')

    # Calculate structured metrics
    weekly_metrics = grouped_week.agg(
        total_trials=('trial_id', 'nunique'),
        violent_trials=('offence_cat', lambda x: x.isin(VIOLENT_CATS).sum()),
        property_trials=('offence_cat', lambda x: x.isin(PROPERTY_CATS).sum()),
        guilty_verdicts=('verdict_cat', lambda x: x.isin(GUILTY_VERDICTS).sum()),
        not_guilty_verdicts=('verdict_cat', lambda x: x.isin(NOT_GUILTY_VERDICTS).sum()),
        death_sentences=('punishment_cat', lambda x: x.isin(DEATH_PUNISH).sum()),
        transport_sentences=('punishment_cat', lambda x: x.isin(TRANSPORT_PUNISH).sum()),
        corporal_sentences=('punishment_cat', lambda x: x.isin(CORPORAL_PUNISH).sum()),
        # Calculate overall fear sentiment
        overall_fear_sentiment=('fear_score', 'mean')
    ).reset_index()

    # Calculate conditional fear sentiment separately
    violent_sentiment = trial_df_scored[trial_df_scored['offence_cat'].isin(VIOLENT_CATS)].groupby('week_date')['fear_score'].mean().reset_index()
    violent_sentiment.rename(columns={'fear_score': 'violent_crime_sentiment'}, inplace=True)

    property_sentiment = trial_df_scored[trial_df_scored['offence_cat'].isin(PROPERTY_CATS)].groupby('week_date')['fear_score'].mean().reset_index()
    property_sentiment.rename(columns={'fear_score': 'property_crime_sentiment'}, inplace=True)

    # Merge conditional sentiments back
    weekly_metrics = pd.merge(weekly_metrics, violent_sentiment, on='week_date', how='left')
    weekly_metrics = pd.merge(weekly_metrics, property_sentiment, on='week_date', how='left')

    # --- Calculate Proportions and Punishment Score ---
    total_trials_denom = weekly_metrics['total_trials'].replace(0, 1)
    guilty_denom = weekly_metrics['guilty_verdicts'].replace(0, 1)
    valid_verdicts_denom = (weekly_metrics['guilty_verdicts'] + weekly_metrics['not_guilty_verdicts']).replace(0, 1)

    weekly_metrics['violent_crime_prop'] = weekly_metrics['violent_trials'] / total_trials_denom
    weekly_metrics['property_crime_prop'] = weekly_metrics['property_trials'] / total_trials_denom
    weekly_metrics['conviction_rate'] = weekly_metrics['guilty_verdicts'] / valid_verdicts_denom
    weekly_metrics['death_sentence_rate'] = weekly_metrics['death_sentences'] / guilty_denom
    weekly_metrics['transport_rate'] = weekly_metrics['transport_sentences'] / guilty_denom

    # Punishment Severity Index
    def calculate_punishment_score(row):
        score = 0
        if row['punishment_cat'] in DEATH_PUNISH: score = 5
        elif row['punishment_cat'] in TRANSPORT_PUNISH: score = 4
        elif row['punishment_cat'] in CORPORAL_PUNISH: score = 3
        return score
    trial_df_scored['punish_score'] = trial_df_scored.apply(calculate_punishment_score, axis=1)
    convicted_df = trial_df_scored[trial_df_scored['verdict_cat'].isin(GUILTY_VERDICTS)]
    weekly_avg_punish_score = convicted_df.groupby('week_date')['punish_score'].mean().reset_index()
    weekly_avg_punish_score.rename(columns={'punish_score': 'avg_punishment_score'}, inplace=True)
    weekly_metrics = pd.merge(weekly_metrics, weekly_avg_punish_score, on='week_date', how='left')

    logger.info(f"Calculated weekly combined metrics for {weekly_metrics.shape[0]} weeks.")

    # Select relevant columns
    metrics_to_merge = weekly_metrics[[
        'week_date', 'total_trials', 'violent_crime_prop', 'property_crime_prop',
        'conviction_rate', 'death_sentence_rate', 'transport_rate', 'avg_punishment_score',
        'overall_fear_sentiment', 'violent_crime_sentiment', 'property_crime_sentiment' # Add new sentiment metrics
    ]]

    # --- Merge with Mortality & Impute ---
    logger.info(f"Merging weekly combined metrics with weekly mortality data...")
    weekly_mortality_df['week_date'] = pd.to_datetime(weekly_mortality_df['week_date'])
    merged_df = pd.merge(weekly_mortality_df, metrics_to_merge, on='week_date', how='outer')
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)

    # Impute missing metrics (often occurs for sentiment if no crimes of that type happened)
    metric_cols_to_impute = [
        'total_trials', 'violent_crime_prop', 'property_crime_prop', 'conviction_rate',
        'death_sentence_rate', 'transport_rate', 'avg_punishment_score',
        'overall_fear_sentiment', 'violent_crime_sentiment', 'property_crime_sentiment'
    ]
    for col in metric_cols_to_impute:
        missing_count = merged_df[col].isnull().sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} weeks have missing data for '{col}'. Imputing with rolling median then 0.")
            merged_df[col] = merged_df[col].fillna(merged_df[col].rolling(4, min_periods=1, center=True).median())
            merged_df[col].fillna(0, inplace=True) # Fill remaining NaNs (esp. for sentiment if no relevant crime type)

    missing_deaths = merged_df['deaths'].isnull().sum()
    if missing_deaths > 0: logger.warning(f"{missing_deaths} weeks have missing mortality data. Imputing with 0."); merged_df['deaths'].fillna(0, inplace=True)

    merged_df['year'] = merged_df['year'].fillna(merged_df['week_date'].dt.isocalendar().year).astype(int)
    merged_df['week_of_year'] = merged_df['week_of_year'].fillna(merged_df['week_date'].dt.isocalendar().week).astype(int)

    # --- Log Transform Deaths ---
    merged_df['log_deaths'] = np.log1p(merged_df['deaths'])
    logger.info("Applied log1p transformation to 'deaths' column -> 'log_deaths'.")

    # --- Create Lagged Features (Example: Lagged Property Crime Sentiment) ---
    lag_weeks = 4
    lag_col_name = 'property_crime_sentiment' # Choose metric to lag based on hypothesis
    feature_lag_col_name = f'{lag_col_name}_lag{lag_weeks}w'
    logger.info(f"Creating lagged feature: '{feature_lag_col_name}' ({lag_weeks} weeks)...")
    merged_df[feature_lag_col_name] = merged_df[lag_col_name].shift(lag_weeks)
    initial_nan_count = merged_df[feature_lag_col_name].isnull().sum()
    if initial_nan_count > 0:
        logger.info(f"Imputing {initial_nan_count} initial NaNs in lagged feature (rolling median then 0).")
        merged_df[feature_lag_col_name] = merged_df[feature_lag_col_name].fillna(merged_df[feature_lag_col_name].rolling(4, min_periods=1, center=True).median())
        merged_df[feature_lag_col_name].fillna(0, inplace=True)

    # --- Standardize Features ---
    features_to_standardize = metric_cols_to_impute + [feature_lag_col_name] # Standardize metrics and the lag
    standardized_col_names = {}
    for col in features_to_standardize:
        if col in merged_df.columns:
            std_col_name = f'{col}_std'
            standardized_col_names[col] = std_col_name
            logger.info(f"Standardizing '{col}' -> '{std_col_name}'.")
            scaler = StandardScaler()
            merged_df[std_col_name] = scaler.fit_transform(merged_df[[col]])
            merged_df[std_col_name] = merged_df[std_col_name].astype(float)
        else: logger.warning(f"Column '{col}' not found for standardization.")

    # --- Prepare Final DataFrame ---
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["week_date"] - merged_df["week_date"].min()).dt.days // 7

    # Final dtypes
    merged_df["log_deaths"] = merged_df["log_deaths"].astype(float)
    merged_df["deaths"] = merged_df["deaths"].astype(float)
    for col in metric_cols_to_impute: merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').astype(float)
    if feature_lag_col_name in merged_df.columns: merged_df[feature_lag_col_name] = merged_df[feature_lag_col_name].astype(float)
    merged_df["year"] = merged_df["year"].astype(str)
    merged_df["week_of_year"] = merged_df["week_of_year"].astype(str)
    merged_df["series_id"] = "London"

    # Define final columns
    final_cols = (
        ["week_date", "time_idx", "deaths", "log_deaths", "year", "week_of_year", "series_id"] +
        metric_cols_to_impute + # Original calculated metrics
        [feature_lag_col_name] + # Lagged metric
        [standardized_col_names.get(col) for col in features_to_standardize if standardized_col_names.get(col)] # Standardized versions
    )
    final_cols_exist = [col for col in final_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols_exist].copy()

    logger.info(f"Shape before final NaN drop and clipping: {merged_df.shape}")
    logger.info(f"Columns before final NaN drop: {merged_df.columns.tolist()}")

    # Drop NaNs based on model features (standardized usually) and target
    critical_cols_for_model = ['time_idx', 'log_deaths'] + [standardized_col_names.get(col) for col in features_to_standardize if standardized_col_names.get(col)]
    critical_cols_for_model_exist = [col for col in critical_cols_for_model if col in merged_df.columns]
    logger.info(f"Dropping NaNs based on columns: {critical_cols_for_model_exist}")
    merged_df.dropna(subset=critical_cols_for_model_exist, inplace=True)
    logger.info(f"Shape after final NaN drop: {merged_df.shape}")

    # --- Clip Ends ---
    if not merged_df.empty:
        weeks_to_clip = 52
        min_idx = merged_df['time_idx'].min()
        max_idx = merged_df['time_idx'].max()
        original_rows = len(merged_df)
        if original_rows > 2 * weeks_to_clip :
            merged_df = merged_df[(merged_df['time_idx'] >= min_idx + weeks_to_clip) &
                                  (merged_df['time_idx'] <= max_idx - weeks_to_clip)].copy()
            logger.info(f"Clipped ends: Removed first/last {weeks_to_clip} weeks. New shape: {merged_df.shape}. ({original_rows - len(merged_df)} rows removed)")
        else: logger.warning(f"Not enough data ({original_rows} rows) to clip {weeks_to_clip} weeks. Skipping.")

        if merged_df.empty:
            logger.error("DataFrame empty after clipping ends.")
            return pd.DataFrame(columns=final_cols_exist)

        if merged_df["time_idx"].max() - merged_df["time_idx"].min() + 1 < WEEKLY_MAX_ENCODER_LENGTH + WEEKLY_MAX_PREDICTION_LENGTH:
             logger.error(f"Insufficient weekly data span ({merged_df.shape[0]} weeks after clipping) for TFT config.")
             return pd.DataFrame(columns=final_cols_exist)
    else:
        logger.error("DataFrame is empty after NaN drop, cannot proceed.")
        return pd.DataFrame(columns=final_cols_exist)

    logger.info(f"Final combined metrics data shape returned: {merged_df.shape}. Time idx: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Final Columns Returned: {merged_df.columns.tolist()}")
    nan_check_after = merged_df.isnull().sum()
    if nan_check_after.any(): logger.warning(f"NaNs still present:\n{nan_check_after[nan_check_after > 0]}")

    return merged_df


@memory.cache
# Use the global AGGREGATION_METHOD set to 'max' above
def aggregate_weekly_sentiment_and_merge(text_df: pd.DataFrame, weekly_mortality_df: pd.DataFrame, agg_method: str = AGGREGATION_METHOD, fear_thresh: float = FEAR_THRESHOLD) -> pd.DataFrame:
    """
    (Cached) Aggregates WEEKLY fear scores from text data based on 'week_date' (using specified method),
    merges with WEEKLY mortality data, log-transforms deaths, creates lagged features, clips ends, and standardizes.
    Returns a weekly DataFrame ready for TFT.
    """
    logger.info(f"Aggregating WEEKLY sentiment using '{agg_method}' and merging with WEEKLY mortality...") # Log reflects the actual method used
    if 'fear_score' not in text_df.columns or 'week_date' not in text_df.columns: raise ValueError("'fear_score','week_date' needed in text_df.")
    if not all(c in weekly_mortality_df.columns for c in ['week_date', 'year', 'week_of_year', 'deaths']): raise ValueError("mortality_df missing cols.")

    text_df_copy = text_df[['week_date', 'fear_score']].copy()
    text_df_copy['fear_score'] = pd.to_numeric(text_df_copy['fear_score'], errors='coerce')
    text_df_copy.dropna(subset=['fear_score', 'week_date'], inplace=True)

    # --- Aggregate Fear Score by WEEK using chosen method ---
    weekly_sentiment = pd.DataFrame()
    feature_col_name = f'fear_score_weekly_{agg_method}'
    logger.info(f"Using aggregation method: '{agg_method}'") # This log is now accurate

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
    weekly_sentiment['week_date'] = pd.to_datetime(weekly_sentiment['week_date'])
    weekly_mortality_df['week_date'] = pd.to_datetime(weekly_mortality_df['week_date'])
    merged_df = pd.merge(weekly_mortality_df, weekly_sentiment, on='week_date', how='outer')
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)

    # --- Handle Missing Data from Merge ---
    missing_sentiment = merged_df[feature_col_name].isnull().sum()
    missing_deaths = merged_df['deaths'].isnull().sum()
    if missing_sentiment > 0: logger.warning(f"{missing_sentiment} weeks have missing sentiment data. Imputing with rolling median.")
    if missing_deaths > 0: logger.warning(f"{missing_deaths} weeks have missing mortality data. Imputing with 0.") # Still imputing raw deaths with 0

    merged_df[feature_col_name] = merged_df[feature_col_name].fillna(merged_df[feature_col_name].rolling(4, min_periods=1, center=True).median())
    merged_df[feature_col_name].fillna(merged_df[feature_col_name].median(), inplace=True)
    merged_df['deaths'].fillna(0, inplace=True)
    # Fill missing year/week carefully after outer merge
    merged_df['year'] = merged_df['year'].fillna(merged_df['week_date'].dt.isocalendar().year)
    merged_df['week_of_year'] = merged_df['week_of_year'].fillna(merged_df['week_date'].dt.isocalendar().week)
    # Ensure integer types where appropriate BEFORE converting to string later if needed
    merged_df['year'] = merged_df['year'].astype(int)
    merged_df['week_of_year'] = merged_df['week_of_year'].astype(int)


    # --- Log Transform Deaths ---
    merged_df['log_deaths'] = np.log1p(merged_df['deaths']) # Use log1p for stability if deaths=0
    logger.info("Applied log1p transformation to 'deaths' column -> 'log_deaths'.")

    # --- Create Lagged Weekly Fear Score ---
    lag_weeks = 4
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

    # --- Prepare for TFT (Weekly Time Index & Dtypes) ---
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["week_date"] - merged_df["week_date"].min()).dt.days // 7

    # Ensure ALL numeric types are correct before converting categoricals
    numeric_cols_to_check = ["time_idx", "deaths", "log_deaths", feature_col_name, feature_lag_col_name, feature_std_col_name, feature_lag_std_col_name]
    for col in numeric_cols_to_check:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce') # Ensure numeric

    # Convert categoricals needed for TFT to string *after* ensuring numeric types are set
    merged_df["year"] = merged_df["year"].astype(str)
    merged_df["week_of_year"] = merged_df["week_of_year"].astype(str)
    merged_df["series_id"] = "London" # This should already be a string

    # Define final columns *including* log_deaths
    final_cols = [
        "week_date", "time_idx", "deaths", "log_deaths", # Ensure log_deaths is here
        "year", "week_of_year", "series_id",
        feature_col_name, feature_lag_col_name,
        feature_std_col_name, feature_lag_std_col_name
    ]
    # Select only the defined final columns, dropping any others
    merged_df = merged_df[final_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

    logger.info(f"Shape before NaN drop and clipping: {merged_df.shape}")
    logger.info(f"Columns before NaN drop: {merged_df.columns.tolist()}") # Log columns
    nan_check_before = merged_df.isnull().sum()
    logger.info(f"NaN check before drop:\n{nan_check_before[nan_check_before > 0]}")

    # Drop any remaining rows with NaNs in critical columns BEFORE clipping
    critical_cols = ['time_idx', 'log_deaths', feature_std_col_name, feature_lag_std_col_name] # Target is log_deaths
    merged_df.dropna(subset=critical_cols, inplace=True)
    logger.info(f"Shape after final NaN drop: {merged_df.shape}")

    # --- Clip Ends (e.g., remove first and last year) ---
    if not merged_df.empty: # Proceed only if dataframe is not empty
        weeks_to_clip = 52 # Remove roughly the first and last year
        min_idx = merged_df['time_idx'].min()
        max_idx = merged_df['time_idx'].max()
        original_rows = len(merged_df)

        # Ensure indices exist before trying to clip
        if original_rows > 2 * weeks_to_clip : # Check if enough data exists to clip
            merged_df = merged_df[(merged_df['time_idx'] >= min_idx + weeks_to_clip) &
                                  (merged_df['time_idx'] <= max_idx - weeks_to_clip)].copy()
            logger.info(f"Clipped ends: Removed first/last {weeks_to_clip} weeks. New shape: {merged_df.shape}. ({original_rows - len(merged_df)} rows removed)")
        else:
            logger.warning(f"Not enough data ({original_rows} rows) to clip {weeks_to_clip} weeks from each end. Skipping clipping.")


        if merged_df.empty:
            logger.error("DataFrame empty after clipping ends. Adjust weeks_to_clip or check data range.")
            # Return empty dataframe with expected columns to avoid downstream errors
            return pd.DataFrame(columns=final_cols)

        # Check sufficiency for TFT *after* clipping
        if merged_df["time_idx"].max() - merged_df["time_idx"].min() + 1 < WEEKLY_MAX_ENCODER_LENGTH + WEEKLY_MAX_PREDICTION_LENGTH:
             logger.error(f"Insufficient weekly data span ({merged_df.shape[0]} weeks after clipping) for TFT config (enc={WEEKLY_MAX_ENCODER_LENGTH}, pred={WEEKLY_MAX_PREDICTION_LENGTH}).")
             # Return empty dataframe with expected columns
             return pd.DataFrame(columns=final_cols)
    else:
        logger.error("DataFrame is empty after NaN drop, cannot proceed.")
        return pd.DataFrame(columns=final_cols) # Return empty dataframe


    logger.info(f"Final weekly data shape returned: {merged_df.shape}. Time idx range: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Final Columns Returned: {merged_df.columns.tolist()}")
    nan_check_after = merged_df.isnull().sum()
    if nan_check_after.any():
         logger.warning(f"NaNs still present after all processing:\n{nan_check_after[nan_check_after > 0]}")

    return merged_df

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
# --- MODIFIED TFT Training Function ---
def train_tft_model(df: pd.DataFrame,
                    # --- ADD new parameter for feature names ---
                    time_varying_reals_cols: List[str], #<<< ADD THIS PARAMETER
                    # --- Keep other parameters ---
                    max_epochs: int = MAX_EPOCHS,
                    batch_size: int = BATCH_SIZE,
                    encoder_length: int = WEEKLY_MAX_ENCODER_LENGTH,
                    pred_length: int = WEEKLY_MAX_PREDICTION_LENGTH,
                    lr: float = LEARNING_RATE,
                    hidden_size: int = HIDDEN_SIZE,
                    attn_heads: int = ATTENTION_HEAD_SIZE,
                    dropout: float = DROPOUT,
                    hidden_cont_size: int = HIDDEN_CONTINUOUS_SIZE,
                    clip_val: float = GRADIENT_CLIP_VAL) -> Tuple[Optional[TemporalFusionTransformer], Optional[pl.Trainer], Optional[torch.utils.data.DataLoader], Optional[TimeSeriesDataSet]]:
    """
    Trains the Temporal Fusion Transformer model on WEEKLY data using log-transformed deaths
    and dynamically specified real-valued features.
    """

    logger.info(f"Setting up WEEKLY TFT model training (Target: log_deaths)...")
    # Use the parameter passed to the function
    logger.info(f" Using real features: {time_varying_reals_cols}")
    logger.info(f" Encoder length: {encoder_length} weeks, Prediction length: {pred_length} weeks")

    # --- Data Cutoff ---
    max_idx = df["time_idx"].max(); training_cutoff = max_idx - pred_length
    min_idx = df["time_idx"].min()
    logger.info(f"Weekly Data Cutoff for Training: time_idx <= {training_cutoff} (Range: {min_idx}-{max_idx})")
    if training_cutoff < min_idx + encoder_length -1:
        logger.error(f"Training cutoff {training_cutoff} doesn't allow full encoder length {encoder_length} from start {min_idx}.")
        return None, None, None, None

    # --- Dtype Check ---
    logger.info("Ensuring correct dtypes before TimeSeriesDataSet...")
    try:
        data_for_tft = df.copy()
        # Combine standard numeric cols with the dynamic list
        required_numeric_cols = ["time_idx", "log_deaths", "deaths"] + time_varying_reals_cols
        required_numeric_cols = list(set(required_numeric_cols)) # Unique columns

        for col in required_numeric_cols:
            if col not in data_for_tft.columns:
                # Log missing columns clearly
                logger.error(f"Column '{col}' specified in features/required list not found in DataFrame columns: {data_for_tft.columns.tolist()}")
                raise ValueError(f"Column '{col}' not found.")
            data_for_tft[col] = pd.to_numeric(data_for_tft[col], errors='coerce')

        categorical_cols = ["series_id", "week_of_year", "year"]
        for col in categorical_cols: data_for_tft[col] = data_for_tft[col].astype(str)

        # Impute NaNs if any crept in
        numeric_cols_to_impute = [col for col in required_numeric_cols if col != 'time_idx']
        if data_for_tft[numeric_cols_to_impute].isnull().any().any():
             nan_counts = data_for_tft[numeric_cols_to_impute].isnull().sum()
             logger.warning(f"NaNs found after casting:\n{nan_counts[nan_counts > 0]}. Imputing with median.")
             for col in numeric_cols_to_impute:
                 data_for_tft[col].fillna(data_for_tft[col].median(), inplace=True)
        logger.info("Dtype check passed.")
    except Exception as e: logger.error(f"Error during dtype check: {e}", exc_info=True); return None, None, None, None

    # --- TimeSeriesDataSet Setup ---
    logger.info("Setting up WEEKLY TimeSeriesDataSet for TFT (Target: log_deaths)...")
    try:
        # Use the dynamic list here, ensuring target is not duplicated
        unknown_reals_for_tft = [col for col in time_varying_reals_cols if col != "log_deaths"]
        logger.info(f"Passing to TimeSeriesDataSet time_varying_unknown_reals: {unknown_reals_for_tft}")

        # Check if all columns in unknown_reals_for_tft actually exist in the dataframe before creating dataset
        missing_tft_cols = [col for col in unknown_reals_for_tft if col not in data_for_tft.columns]
        if missing_tft_cols:
            logger.error(f"Columns specified for TFT `time_varying_unknown_reals` are missing from the dataframe: {missing_tft_cols}")
            raise ValueError("Missing columns required for TFT dataset.")

        training_dataset = TimeSeriesDataSet(
            data_for_tft[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="log_deaths",
            group_ids=["series_id"],
            max_encoder_length=encoder_length, max_prediction_length=pred_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["week_of_year"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=["year"],
            # Use the filtered dynamic list
            time_varying_unknown_reals=unknown_reals_for_tft,
            add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=True,
            categorical_encoders={"year": NaNLabelEncoder(add_nan=True), "week_of_year": NaNLabelEncoder(add_nan=True)}
        )

        # --- Dataloader and Model Training ---
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
            dropout=dropout, hidden_continuous_size=hidden_cont_size, loss=loss_metric, log_interval=50,
            optimizer="adam", reduce_on_plateau_patience=5,
        )
        logger.info(f"TFT model parameters: {tft.size()/1e6:.1f} million")
    except Exception as e: logger.error(f"Error initializing TFT: {e}", exc_info=True); return None, None, val_dataloader, validation_dataset

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accelerator, devices = ('cpu', 1)
    logger.info(f"Configuring Trainer (Accelerator: {accelerator}, Devices: {devices})...")
    from pytorch_lightning.loggers import TensorBoardLogger
    # Updated logger name for combined metrics
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name="tft_combined_metrics_weekly_log_target")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=devices, gradient_clip_val=clip_val, callbacks=[lr_monitor, early_stop_callback], logger=tb_logger, enable_progress_bar=True)

    logger.info("Starting TFT model training (Weekly, Target: log_deaths, Features: Combined Metrics)...")
    start_train_time = time.time()
    try:
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        logger.info(f"TFT training finished in {(time.time() - start_train_time)/60:.2f} minutes.")
        best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from checkpoint: {best_model_path}")
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=DEVICE)
            return best_tft, trainer, val_dataloader, validation_dataset
        else:
            logger.warning("Best checkpoint not found/saved.")
            last_model = trainer.model if hasattr(trainer, 'model') else tft
            if last_model: last_model.to(DEVICE)
            return last_model, trainer, val_dataloader, validation_dataset # Return last model state if no checkpoint
    except Exception as e:
        logger.error(f"Error during TFT fitting: {e}", exc_info=True)
        last_model = trainer.model if hasattr(trainer, 'model') else tft # Try to return model even if fit fails
        if last_model: last_model.to(DEVICE)
        return last_model, trainer, val_dataloader, validation_dataset

# --- evaluate_model function (UPDATED FOR WEEKLY & LOG TRANSFORM) ---
def evaluate_model(model: TemporalFusionTransformer, dataloader: torch.utils.data.DataLoader, dataset: TimeSeriesDataSet, plot_dir: str) -> Dict[str, float]:
    """Evaluates TFT model on log_deaths, returns metrics on original death scale, saves plots."""
    logger.info("Evaluating model performance on validation set (Weekly, Target: log_deaths)...")
    results = {}
    if model is None or dataloader is None or len(dataloader) == 0: logger.error("Model/Dataloader missing for eval."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
    try: eval_device = next(model.parameters()).device
    except Exception: eval_device = torch.device(DEVICE); model.to(eval_device)
    logger.info(f"Evaluation device: {eval_device}")

    actuals_log_list, preds_log_list = [], [] # Store log-scale values from model
    with torch.no_grad():
        for x, y in iter(dataloader):
            x = {k: v.to(eval_device) for k, v in x.items() if isinstance(v, torch.Tensor)}
            target_log = y[0].to(eval_device) # y[0] is log_deaths
            preds = model(x)["prediction"] # Predictions are also on log_deaths scale
            preds_log_list.append(preds.cpu()) # Store full prediction tensor (p10, p50, p90)
            actuals_log_list.append(target_log.cpu())

    if not preds_log_list: logger.error("No predictions collected."); return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    # Concatenate tensors (keep quantiles for now)
    actuals_log_all = torch.cat(actuals_log_list).numpy() # Shape: (n_samples, pred_length)
    preds_log_all = torch.cat(preds_log_list).numpy()    # Shape: (n_samples, pred_length, n_quantiles)

    # Use median prediction (index 1) and flatten for metrics
    actuals_log_flat = actuals_log_all.flatten()
    preds_log_median_flat = preds_log_all[:, :, 1].flatten() # Index 1 corresponds to 0.5 quantile (median)

    min_len_m = min(len(actuals_log_flat), len(preds_log_median_flat))
    if len(actuals_log_flat) != len(preds_log_median_flat):
        logger.warning(f"Metric length mismatch: Truncate.");
        preds_log_median_flat=preds_log_median_flat[:min_len_m]
        actuals_log_flat=actuals_log_flat[:min_len_m]

    # *** Inverse transform to original death scale for MAE/MSE ***
    actuals_orig_flat = np.expm1(actuals_log_flat)
    preds_orig_median_flat = np.expm1(preds_log_median_flat)
    # Ensure non-negative predictions after inverse transform
    preds_orig_median_flat = np.maximum(0, preds_orig_median_flat)

    # Calculate metrics on original scale
    val_mae = mean_absolute_error(actuals_orig_flat, preds_orig_median_flat)
    val_mse = mean_squared_error(actuals_orig_flat, preds_orig_median_flat)
    # SMAPE can be calculated on original or log scale (less sensitive to scale) - let's use original
    denominator = (np.abs(actuals_orig_flat) + np.abs(preds_orig_median_flat)) / 2.0
    val_smape = np.mean(np.abs(preds_orig_median_flat - actuals_orig_flat) / np.where(denominator == 0, 1, denominator)) * 100
    results = {"MAE": val_mae, "MSE": val_mse, "SMAPE": val_smape}
    logger.info(f"[Validation Metrics (Original Scale)] MAE={val_mae:.3f} MSE={val_mse:.3f} SMAPE={val_smape:.3f}%")

    # --- Plotting Section (Using predict for aligned data, show ORIGINAL scale) ---
    logger.info("Generating weekly evaluation plots (showing original death scale)...")
    try:
        # We need the original 'deaths' corresponding to the log_deaths target for plotting actuals
        # Get predictions and raw data using predict
        prediction_output = model.predict(dataloader, mode="raw", return_x=True, return_index=True)
        preds_dict = None; x_dict = None; index_df = None
        if isinstance(prediction_output, (list, tuple)) and len(prediction_output) >= 1:
            if isinstance(prediction_output[0], dict): preds_dict = prediction_output[0] # Contains 'prediction' on log scale
            if len(prediction_output) > 1 and isinstance(prediction_output[1], dict): x_dict = prediction_output[1] # Contains inputs like 'decoder_target' (log_deaths)
            if len(prediction_output) > 2 and isinstance(prediction_output[2], pd.DataFrame): index_df = prediction_output[2] # Contains 'time_idx'
        elif isinstance(prediction_output, dict): preds_dict = prediction_output # Less common return format now
        if preds_dict is None or x_dict is None or index_df is None: logger.error(f"Predict unpack fail. Skip plots."); return results

        # Predictions are log scale
        preds_log_tensor = preds_dict['prediction'].cpu() # Shape: (n_samples, pred_length, n_quantiles)

        # Get ACTUAL log_deaths used by the model during prediction
        actuals_log_tensor = x_dict['decoder_target'].cpu() # Shape: (n_samples, pred_length)

        # Extract time_idx
        time_idx_flat = index_df['time_idx'].values

        # Flatten log predictions (p10, p50, p90)
        preds_log_p10_flat = preds_log_tensor[:, :, 0].flatten().numpy()
        preds_log_p50_flat = preds_log_tensor[:, :, 1].flatten().numpy()
        preds_log_p90_flat = preds_log_tensor[:, :, 2].flatten().numpy()
        actuals_log_flat_plot = actuals_log_tensor.flatten().numpy()

        # Check lengths
        n_preds = len(preds_log_p50_flat); n_actuals = len(actuals_log_flat_plot); n_time = len(time_idx_flat)
        logger.debug(f"Plot shapes - Preds: {n_preds}, Actuals: {n_actuals}, Time: {n_time}")
        if not (n_preds == n_actuals == n_time):
            logger.warning(f"Plot length mismatch! Truncate."); min_len_plot = min(n_preds, n_actuals, n_time)
            if min_len_plot == 0: logger.error("Zero length plot data. Skip plots."); return results
            preds_log_p10_flat=preds_log_p10_flat[:min_len_plot]; preds_log_p50_flat=preds_log_p50_flat[:min_len_plot]; preds_log_p90_flat=preds_log_p90_flat[:min_len_plot]
            actuals_log_flat_plot=actuals_log_flat_plot[:min_len_plot]; time_idx_flat=time_idx_flat[:min_len_plot]

        # *** Inverse transform ACTUALS and PREDICTIONS for plotting ***
        actuals_orig_flat_plot = np.expm1(actuals_log_flat_plot)
        p10_orig_flat = np.maximum(0, np.expm1(preds_log_p10_flat))
        p50_orig_flat = np.maximum(0, np.expm1(preds_log_p50_flat))
        p90_orig_flat = np.maximum(0, np.expm1(preds_log_p90_flat))

        # Sort by time_idx for plotting
        sort_indices = np.argsort(time_idx_flat)
        time_idx_sorted=time_idx_flat[sort_indices]
        actuals_sorted=actuals_orig_flat_plot[sort_indices] # Plot original scale actuals
        p10_sorted=p10_orig_flat[sort_indices]; p50_sorted=p50_orig_flat[sort_indices]; p90_sorted=p90_orig_flat[sort_indices]

        # --- Generate Plots (Original Scale) ---
        try: start_date_dt = dataset.data["raw"]["week_date"].min(); x_label = f"Time Index (Weeks since {start_date_dt:%Y-%m-%d})"
        except Exception: x_label = "Time Index (Weeks)"

        plt.figure(figsize=(15, 7))
        plt.plot(time_idx_sorted, actuals_sorted, label="Actual Deaths", marker='.', linestyle='-', alpha=0.7, color='black', markersize=3, linewidth=0.8)
        plt.plot(time_idx_sorted, p50_sorted, label="Predicted Median (p50)", linestyle='--', alpha=0.9, color='tab:orange', linewidth=1.2)
        plt.fill_between(time_idx_sorted, p10_sorted, p90_sorted, color='tab:orange', alpha=0.3, label='p10-p90 Quantiles')
        plt.title(f"TFT Aggregated Forecast vs Actuals (Validation Set - Weekly, Original Scale)\nMAE={val_mae:.2f}, SMAPE={val_smape:.2f}%", fontsize=14)
        plt.xlabel(x_label, fontsize=12); plt.ylabel("Weekly Deaths (Original Scale)", fontsize=12) # Updated label
        plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        plot_file = os.path.join(plot_dir, "tft_val_forecast_aggregated_weekly_original_scale.png"); plt.savefig(plot_file); logger.info(f"Saved aggregated forecast plot to {plot_file}"); plt.close()

        # Residual Plot (on original scale)
        residuals = actuals_sorted - p50_sorted
        plt.figure(figsize=(10, 6)); plt.scatter(p50_sorted, residuals, alpha=0.3, s=15, color='tab:blue', edgecolors='k', linewidth=0.5)
        plt.axhline(0, color='red', linestyle='--', linewidth=1); plt.title('Residual Plot (Actuals - Median Predictions, Original Scale)', fontsize=14)
        plt.xlabel('Predicted Median Deaths (Original Scale)', fontsize=12); plt.ylabel('Residuals (Original Scale)', fontsize=12); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        save_path_res = os.path.join(plot_dir, "residuals_weekly_original_scale_plot.png"); plt.savefig(save_path_res); logger.info(f"Saved residual plot to {save_path_res}"); plt.close()

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
    """Main script execution flow using Old Bailey (Structured + Text) and Weekly data."""
    logger.info("--- Starting Historical Analysis Script (Old Bailey - Combined Approach) ---")
    script_start_time = time.time()
    final_df = None; best_model = None; trainer = None; val_dl = None; val_ds = None

    try:
        os.makedirs(PLOT_DIR, exist_ok=True); logger.info(f"Plots directory: {PLOT_DIR}")
        lag_weeks = 4

        # --- Step 1: Parse Old Bailey Data (Structured + Text) ---
        logger.info("--- Step 1: Parsing Old Bailey Sessions Papers (Structured Data + Text) ---")
        # This function should return ['week_date', 'doc_id', 'trial_id', 'text', 'offence_cat', 'verdict_cat', 'punishment_cat']
        ob_df_parsed = parse_old_bailey_papers(ob_dir=OLD_BAILEY_DIR, start_year=START_YEAR, end_year=END_YEAR)
        if not isinstance(ob_df_parsed, pd.DataFrame) or ob_df_parsed.empty: raise ValueError("Step 1 Failed: No Old Bailey trial data found/parsed.")
        logger.info(f"Parsed {len(ob_df_parsed)} trial accounts.")

        # --- Step 2: Preprocess Trial Text ---
        logger.info("--- Step 2: Preprocessing Old Bailey Trial Text ---")
        USE_SYMSPELL_CORRECTION = False
        # This function adds 'processed_text' column
        ob_df_processed = preprocess_text_dataframe(ob_df_parsed, text_col='text', use_symspell=USE_SYMSPELL_CORRECTION)
        if ob_df_processed.empty: raise ValueError("Step 2 Failed: Text preprocessing resulted in empty DataFrame.")
        # Keep relevant structured columns along with processed text
        cols_to_keep = ['week_date', 'doc_id', 'trial_id', 'processed_text', 'offence_cat', 'verdict_cat', 'punishment_cat']
        # Ensure all columns exist before selection
        cols_to_keep_exist = [col for col in cols_to_keep if col in ob_df_processed.columns]
        ob_df_processed = ob_df_processed[cols_to_keep_exist]
        del ob_df_parsed # Free memory

        # --- Step 3: Calculate Fear Scores per Trial ---
        logger.info("--- Step 3: Calculating MacBERTh Scores per Trial ---")
        # This function adds 'fear_score' column
        ob_df_scored = calculate_fear_scores_dataframe(ob_df_processed, text_col='processed_text', batch_size=BATCH_SIZE // 2)
        if 'fear_score' not in ob_df_scored.columns: raise ValueError("Step 3 Failed: Fear scoring failed.")
        logger.info(f"Scored {len(ob_df_scored)} trials.")
        del ob_df_processed # Free memory

        # --- Step 4: Load Weekly Mortality ---
        logger.info("--- Step 4: Loading and Processing Weekly Mortality Data ---")
        weekly_mortality_df = load_and_aggregate_weekly_mortality(file_path=COUNTS_FILE, start_year=START_YEAR, end_year=END_YEAR)
        if weekly_mortality_df.empty: raise ValueError("Step 4 Failed: No weekly mortality data found.")

        # --- Step 5: Aggregate Combined Metrics (Weekly) & Merge ---
        logger.info(f"--- Step 5: Aggregating Weekly Combined Metrics and Merging ---")
        # This function uses the trial-level scores and structured data
        final_df = aggregate_combined_metrics(ob_df_scored, weekly_mortality_df)
        if final_df.empty: raise ValueError("Step 5 Failed: Merging/Clipping resulted in empty DataFrame.")
        del ob_df_scored; del weekly_mortality_df # Free memory

        # Save final weekly data
        try:
            # Make filename more descriptive
            final_df.to_csv(os.path.join(PLOT_DIR, f"final_weekly_data_combined_metrics_logdeaths.csv"), index=False)
            logger.info("Saved final weekly data with combined metrics.")
        except Exception as e: logger.warning(f"Could not save final data CSV: {e}")

        # --- Step 5.5: Generate Data Exploration Plots (Weekly) ---
        logger.info("--- Step 5.5: Generating Data Exploration Plots (Weekly - Combined Metrics) ---")
        # Define columns based on the new aggregation function's output
        prop_sent_col = 'property_crime_sentiment'; viol_sent_col = 'violent_crime_sentiment'; overall_sent_col = 'overall_fear_sentiment'
        viol_prop_col = 'violent_crime_prop'; prop_prop_col = 'property_crime_prop'; conv_rate_col = 'conviction_rate'
        punish_score_col = 'avg_punishment_score'
        lag_prop_sent_col = 'property_crime_sentiment_lag4w' # Example lagged column (adjust if lagging different col)
        prop_sent_std_col = 'property_crime_sentiment_std' # Example standardized column (adjust if standardizing different col)

        if final_df is not None and not final_df.empty:
            plot_time_series(final_df, 'week_date', 'deaths', 'Weekly Deaths Over Time (London Bills - Original Scale)', 'Total Deaths', 'deaths_weekly_timeseries_original.png', PLOT_DIR)
            if 'log_deaths' in final_df.columns: plot_time_series(final_df, 'week_date', 'log_deaths', 'Weekly Log(Deaths+1) Over Time', 'Log(Deaths+1)', 'deaths_weekly_timeseries_log.png', PLOT_DIR)
            else: logger.error("Column 'log_deaths' not found for plotting.")

            # Plot new sentiment metrics
            if prop_sent_col in final_df.columns: plot_time_series(final_df, 'week_date', prop_sent_col, 'Weekly Sentiment (Property Crimes)', 'Avg. Fear Score', 'property_crime_sentiment_timeseries.png', PLOT_DIR)
            if viol_sent_col in final_df.columns: plot_time_series(final_df, 'week_date', viol_sent_col, 'Weekly Sentiment (Violent Crimes)', 'Avg. Fear Score', 'violent_crime_sentiment_timeseries.png', PLOT_DIR)
            if overall_sent_col in final_df.columns: plot_time_series(final_df, 'week_date', overall_sent_col, 'Weekly Sentiment (Overall)', 'Avg. Fear Score', 'overall_fear_sentiment_timeseries.png', PLOT_DIR)

            # Plot key structured metrics
            if viol_prop_col in final_df.columns: plot_time_series(final_df, 'week_date', viol_prop_col, 'Weekly Violent Crime Proportion', 'Proportion of Trials', 'violent_crime_prop_timeseries.png', PLOT_DIR)
            if prop_prop_col in final_df.columns: plot_time_series(final_df, 'week_date', prop_prop_col, 'Weekly Property Crime Proportion', 'Proportion of Trials', 'property_crime_prop_timeseries.png', PLOT_DIR)
            if conv_rate_col in final_df.columns: plot_time_series(final_df, 'week_date', conv_rate_col, 'Weekly Conviction Rate', 'Rate', 'conviction_rate_timeseries.png', PLOT_DIR)
            if punish_score_col in final_df.columns: plot_time_series(final_df, 'week_date', punish_score_col, 'Weekly Avg. Punishment Score', 'Avg Score', 'punishment_score_timeseries.png', PLOT_DIR)

            # Dual axis: Log Deaths vs Property Crime Sentiment (Standardized) - Adjust col names if needed
            # Ensure the standardized column exists from the aggregation step
            if 'log_deaths' in final_df.columns and prop_sent_std_col in final_df.columns:
                 plot_dual_axis(final_df, 'week_date', 'log_deaths', prop_sent_std_col, 'Log(Deaths+1)', 'Std. Property Crime Sent.', 'Log(Deaths+1) vs Std. Property Crime Sentiment', 'logdeaths_vs_prop_crime_sent_std_dual_axis.png', PLOT_DIR)
            elif 'log_deaths' in final_df.columns and prop_sent_col in final_df.columns:
                 logger.warning(f"Standardized column '{prop_sent_std_col}' not found, plotting original '{prop_sent_col}' on dual axis.")
                 plot_dual_axis(final_df, 'week_date', 'log_deaths', prop_sent_col, 'Log(Deaths+1)', 'Property Crime Sent.', 'Log(Deaths+1) vs Property Crime Sentiment', 'logdeaths_vs_prop_crime_sent_dual_axis.png', PLOT_DIR)

            # Scatter: Log Deaths vs Lagged Property Crime Sentiment - Adjust col names if needed
            if 'log_deaths' in final_df.columns and lag_prop_sent_col in final_df.columns:
                plot_scatter_fear_vs_deaths(final_df, lag_prop_sent_col, 'log_deaths', f'Log(Deaths+1) vs Lagged ({lag_weeks}w) Property Crime Sentiment', f'scatter_logdeaths_vs_{lag_prop_sent_col}.png', PLOT_DIR)
            elif 'log_deaths' in final_df.columns and prop_sent_col in final_df.columns:
                 logger.warning(f"Lagged column '{lag_prop_sent_col}' not found, plotting scatter with original '{prop_sent_col}'.")
                 plot_scatter_fear_vs_deaths(final_df, prop_sent_col, 'log_deaths', 'Log(Deaths+1) vs Property Crime Sentiment', f'scatter_logdeaths_vs_{prop_sent_col}.png', PLOT_DIR)


            plot_weekly_boxplot(final_df, 'deaths', 'Weekly Distribution of Deaths (by Week of Year - Original Scale)', 'boxplot_deaths_weekly_original.png', PLOT_DIR)
        else: logger.warning("final_df empty, skipping data plots.")

        # === Step 6: Train and Evaluate TFT Model (Weekly) ===
        logger.info("--- Step 6: Training and Evaluating TFT Model (Weekly - Target: log_deaths, Features: Combined Metrics) ---")
        if final_df is not None and not final_df.empty:
            # *** Define the STANDARDIZED metrics to use as features for TFT ***
            # Choose a subset based on hypotheses and initial analysis
            # Example: Use standardized sentiment metrics and conviction rate + lags
            # These are the *base* names before '_std' is added
            tft_feature_base_cols = [
                'overall_fear_sentiment', 'violent_crime_sentiment', 'property_crime_sentiment', # Sentiment features
                'violent_crime_prop', 'property_crime_prop', 'conviction_rate', 'avg_punishment_score', # Structured features
                lag_prop_sent_col # Include the lagged version base name
            ]
            # Create the list of standardized names expected from aggregate_combined_metrics
            tft_real_features_std = [col + '_std' for col in tft_feature_base_cols if col + '_std' in final_df.columns]

            if not tft_real_features_std:
                logger.error("No standardized feature columns found to use for TFT. Skipping training.")
                best_model, trainer, val_dl, val_ds = None, None, None, None
            else:
                logger.info(f"Using these standardized features as time_varying_unknown_reals for TFT: {tft_real_features_std}")
                # Ensure train_tft_model is adapted to take this list dynamically
                # (Using the modified train_tft_model from previous response)
                best_model, trainer, val_dl, val_ds = train_tft_model(
                    df=final_df,
                    time_varying_reals_cols=tft_real_features_std, # Pass the list of existing std columns
                    encoder_length=WEEKLY_MAX_ENCODER_LENGTH,
                    pred_length=WEEKLY_MAX_PREDICTION_LENGTH,
                    # Pass other TFT hyperparameters from config if needed
                    max_epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE,
                    hidden_size=HIDDEN_SIZE,
                    attn_heads=ATTENTION_HEAD_SIZE,
                    dropout=DROPOUT,
                    hidden_cont_size=HIDDEN_CONTINUOUS_SIZE,
                    clip_val=GRADIENT_CLIP_VAL
                )

            if best_model and val_dl and val_ds:
                logger.info("--- Step 6.5: Evaluating Best TFT Model ---")
                eval_metrics = evaluate_model(best_model, val_dl, val_ds, plot_dir=PLOT_DIR)
                logger.info(f"Final Validation Metrics: {eval_metrics}")
            # Adjust message if training was skipped due to lack of features
            elif not tft_real_features_std and not final_df.empty:
                 logger.warning("TFT Training skipped due to missing standardized features.")
            else: logger.error("TFT Model training failed or evaluation components missing.")
        else: logger.error("Final DataFrame empty, cannot train TFT model.")

        # === Step 7: Interpretation & Granger Causality (Weekly) ===
        logger.info("--- Step 7: Model Interpretation & Granger Causality (Weekly - Using Combined Metrics) ---")
        if final_df is not None and not final_df.empty:
            # Interpretation needs model trained on these features
            if best_model and val_dl:
                logger.info("--- Running TFT Interpretation ---")
                interpret_tft(best_model, val_dl, PLOT_DIR)
            else:
                logger.warning("Skipping TFT interpretation as model was not trained or available.")

            logger.info("--- Running Granger Causality Tests (Weekly - Using log_deaths & Combined Metrics) ---")
            granger_results = {}
            max_weekly_lag = 8
            target_col_granger = 'log_deaths'

            # Test the NON-standardized metrics against log_deaths
            # Adjust column names to match those created in aggregate_combined_metrics
            metrics_to_test_granger = {
                'overall_fear_sentiment': 'Overall Sentiment',
                'violent_crime_sentiment': 'Violent Crime Sent.',
                'property_crime_sentiment': 'Property Crime Sent.',
                 lag_prop_sent_col: f'Lag {lag_weeks}w Property Sent.', # Use the variable holding the lagged name
                'violent_crime_prop': 'Violent Crime Prop',
                'property_crime_prop': 'Property Crime Prop',
                'conviction_rate': 'Conviction Rate',
                'avg_punishment_score': 'Avg Punishment Score'
            }

            for metric_col, metric_label in metrics_to_test_granger.items():
                # Check if BOTH the target and the metric column exist
                if target_col_granger in final_df.columns and metric_col in final_df.columns:
                    logger.info(f"Granger tests: '{metric_col}' vs '{target_col_granger}'")
                    desc1 = f"'{metric_label}' -> 'Log(Deaths)'"
                    granger_results[desc1] = run_granger_causality(final_df, metric_col, target_col_granger, max_lag=max_weekly_lag)
                    desc2 = f"'Log(Deaths)' -> '{metric_label}'"
                    granger_results[desc2] = run_granger_causality(final_df, target_col_granger, metric_col, max_lag=max_weekly_lag)
                else:
                    logger.warning(f"Skipping Granger for '{metric_label}' vs 'Log(Deaths)' - one or both columns missing ({metric_col}, {target_col_granger}).")

            # Plot combined results if any tests ran
            if granger_results:
                plot_granger_causality_results(granger_results, title_prefix=f"Weekly Combined Metrics vs Log(Deaths+1)", plot_dir=PLOT_DIR)
            else:
                logger.warning("No valid Granger results to plot.")

        else: logger.warning("Final DataFrame empty, skipping interpretation & Granger.")

    # --- Error Handling & Finish ---
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