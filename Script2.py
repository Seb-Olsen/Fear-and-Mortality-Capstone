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
from tqdm.auto import tqdm
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

# Word lists for targeted sentiment
DISEASE_FEAR_WORDS = ["pestilence", "plague", "contagion", "infection", "epidemic", "pox", "sickness", "disease", "illness", "malady", "distemper", "smallpox", "sweating"]
HARDSHIP_WORDS = ["poor", "poverty", "necessity", "distress", "hardship", "starve", "desperate", "ruin", "beggar", "vagran", "hunger", "want"]
VIOLENCE_SEVERITY_WORDS = ["kill", "murder", "murther", "wound", "stab", "pistol", "weapon", "sword", "hanger", "deadly", "death", "slay", "violence"]
OVERALL_DISTRESS_WORDS = DISEASE_FEAR_WORDS # Use the original broad list (assuming FEAR_WORDS is defined)

# Define the LAG_WEEKS constant
LAG_WEEKS = 4

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
LEARNING_RATE = 0.0005
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

# === Fear Scoring using MacBERTh (MODIFIED FOR MULTIPLE SENTIMENTS) ===
class MacBERThSentimentScorer:
    _instance = None
    _model_name = MACBERTH_MODEL_NAME
    # Define the word lists and corresponding score names HERE
    _word_lists = {
        'disease_sentiment': DISEASE_FEAR_WORDS,
        'hardship_sentiment': HARDSHIP_WORDS
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info(f"Creating MacBERThSentimentScorer instance...")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if self._initialized:
            return
        logger.info(f"Initializing MacBERTh model for embedding: {self._model_name}...")
        self.device = device if device else DEVICE
        logger.info(f"MacBERTh on device: {self.device}")
        self.reference_vectors = {} # Dictionary to hold reference vectors

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name).to(self.device)
            self.model.eval()

            logger.info("Calculating average reference vectors for each sentiment type...")
            with torch.no_grad():
                # Iterate through the specified word lists
                for score_name, word_list in self._word_lists.items():
                    if not word_list:
                        logger.warning(f"Word list for '{score_name}' is empty. Skipping.")
                        self.reference_vectors[score_name] = None
                        continue
                    valid_word_list = [str(w) for w in word_list if isinstance(w, str) and w]
                    if not valid_word_list:
                         logger.warning(f"Valid word list for '{score_name}' is empty after filtering. Skipping.")
                         self.reference_vectors[score_name] = None
                         continue

                    inputs = self.tokenizer(valid_word_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
                    avg_vector = np.mean(embeddings, axis=0).reshape(1, -1)
                    self.reference_vectors[score_name] = avg_vector
                    logger.info(f" - Calculated reference vector for '{score_name}' (shape: {avg_vector.shape}).")

            self._initialized = True
            logger.info("MacBERThSentimentScorer initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize MacBERTh scorer: {e}", exc_info=True)
            raise

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def calculate_sentiment_scores(self, texts: List[str], batch_size: int = 32) -> Dict[str, List[float]]:
        """Calculates multiple sentiment scores for a list of texts."""
        if not self._initialized: raise RuntimeError("Scorer not initialized.")
        if not texts: return {score_name: [] for score_name in self.reference_vectors}

        num_texts = len(texts)
        all_scores = {score_name: [] for score_name in self.reference_vectors if self.reference_vectors.get(score_name) is not None} # Use .get() for safety
        valid_score_names = list(all_scores.keys())
        if not valid_score_names:
            logger.warning("No valid reference vectors calculated. Returning empty scores dict.")
            return {}

        logger.info(f"Calculating {len(valid_score_names)} types of sentiment scores for {num_texts} texts...")
        log_interval = max(1, (num_texts // batch_size) // 10) if batch_size > 0 else num_texts # Avoid division by zero

        # Wrap the range with tqdm for a progress bar
        for i in tqdm(range(0, num_texts, batch_size), desc="Calculating Sentiment Scores"): # <<<--- WRAP HERE
            batch_texts = texts[i : i + batch_size]
            valid_batch_texts = [str(t) if t else "" for t in batch_texts]
            try:
                inputs = self.tokenizer(valid_batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
                for score_name in valid_score_names:
                    ref_vector = self.reference_vectors[score_name]
                    similarities = cosine_similarity(embeddings, ref_vector)
                    all_scores[score_name].extend(similarities.flatten().tolist())
            except Exception as e:
                 logger.error(f"Error processing batch starting at index {i}: {e}. Appending zeros.")
                 batch_len = len(valid_batch_texts)
                 for score_name in valid_score_names:
                     all_scores[score_name].extend([0.0] * batch_len)

        for score_name in valid_score_names:
            if len(all_scores[score_name]) != num_texts:
                logger.error(f"Score length mismatch '{score_name}'. Padding."); all_scores[score_name].extend([0.0] * (num_texts - len(all_scores[score_name])))
        logger.info("Sentiment score calculation complete.")
        return all_scores

@memory.cache
def calculate_sentiment_scores_dataframe(df: pd.DataFrame, text_col: str = "text", batch_size: int = 32) -> pd.DataFrame:
    """Calculates multiple MacBERTh sentiment scores for the text column."""
    logger.info(f"Calculating multiple MacBERTh sentiment scores for '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")

    df_copy = df.copy()
    # Ensure text column is string and handle NaNs
    df_copy[text_col] = df_copy[text_col].astype(str).fillna('')

    scorer = MacBERThSentimentScorer() # Initialize the multi-score scorer
    texts_to_score = df_copy[text_col].tolist()

    # Get the dictionary of scores {score_name: [list_of_scores]}
    sentiment_scores_dict = scorer.calculate_sentiment_scores(texts_to_score, batch_size=batch_size)

    # Add each score list as a new column to the DataFrame
    score_cols_added = []
    for score_name, scores_list in sentiment_scores_dict.items():
        if scores_list: # Only add if scores were calculated
             # Ensure score name doesn't clash with existing columns if df_copy is reused
             if score_name in df_copy.columns:
                 logger.warning(f"Column '{score_name}' already exists. Overwriting.")
             df_copy[score_name] = scores_list
             score_cols_added.append(score_name) # Keep track of added columns
             logger.info(f"Added sentiment scores for '{score_name}'. Stats: Min={np.min(scores_list):.3f}, Max={np.max(scores_list):.3f}, Mean={np.mean(scores_list):.3f}, Std={np.std(scores_list):.3f}")
        else:
             logger.warning(f"No scores calculated for '{score_name}', column not added.")

    logger.info("Multiple sentiment scoring complete.")
    # Return relevant identifier columns along with the new scores
    # Ensure identifiers exist in the input df
    id_cols = ['week_date', 'doc_id', 'trial_id']
    valid_id_cols = [col for col in id_cols if col in df_copy.columns]
    final_cols = valid_id_cols + score_cols_added
    return df_copy[final_cols]

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
def aggregate_weekly_combined_metrics(
    structured_df: pd.DataFrame,  # Output from parse_old_bailey_papers
    sentiment_df: pd.DataFrame,  # Output from calculate_sentiment_scores_dataframe
    weekly_mortality_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    (Cached) Aggregates weekly structured trial metrics AND trial-level sentiment scores,
    applies filtering/smoothing, merges with mortality data, log-transforms deaths,
    creates lagged features, clips ends, and standardizes.
    Returns a weekly DataFrame ready for analysis/TFT.
    """
    logger.info(f"Aggregating WEEKLY combined structured & sentiment metrics...")
    # --- Input Checks ---
    required_structured_cols = ['week_date', 'trial_id', 'offence_cat', 'verdict_cat', 'punishment_cat']
    # Dynamically get expected sentiment columns from the sentiment_df input
    expected_sentiment_cols = ['week_date', 'trial_id'] + [col for col in sentiment_df.columns if col.endswith('_sentiment')]
    required_mortality_cols = ['week_date', 'year', 'week_of_year', 'deaths']

    if not all(c in structured_df.columns for c in required_structured_cols):
         raise ValueError(f"structured_df missing required columns. Need: {required_structured_cols}. Found: {structured_df.columns.tolist()}")
    if not all(c in sentiment_df.columns for c in expected_sentiment_cols):
         # Allow proceeding if sentiment_df is empty but has right id cols
         if not sentiment_df.empty or not all(c in sentiment_df.columns for c in ['week_date', 'trial_id']):
             raise ValueError(f"sentiment_df missing expected columns. Need: {expected_sentiment_cols}. Found: {sentiment_df.columns.tolist()}")
         else:
             logger.warning("Sentiment DataFrame is empty but has identifier columns. Proceeding without sentiment scores.")
             sentiment_cols_to_agg = [] # No sentiment cols to process
    else:
        sentiment_cols_to_agg = [col for col in expected_sentiment_cols if col not in ['week_date', 'trial_id']]

    if not all(c in weekly_mortality_df.columns for c in required_mortality_cols):
        raise ValueError("mortality_df missing required columns.")

    # Convert dates
    structured_df['week_date'] = pd.to_datetime(structured_df['week_date'])
    sentiment_df['week_date'] = pd.to_datetime(sentiment_df['week_date'])
    weekly_mortality_df['week_date'] = pd.to_datetime(weekly_mortality_df['week_date'])

    # --- Aggregate Structured Metrics (Filter, Smooth) ---
    VIOLENT_CATS = ['violenttheft', 'kill', 'sexual']; PROPERTY_CATS = ['theft', 'deception', 'damage', 'royaloffences']
    GUILTY_VERDICTS = ['guilty']; NOT_GUILTY_VERDICTS = ['notguilty', 'unknown']
    DEATH_PUNISH = ['death']; TRANSPORT_PUNISH = ['transport']; CORPORAL_PUNISH = ['corporal', 'miscpunish']
    MIN_TRIALS_PER_WEEK = 3; SMOOTHING_WINDOW = 4

    weekly_counts = structured_df.groupby('week_date').agg(
        total_trials = ('trial_id', 'nunique'),
        violent_trials = ('offence_cat', lambda x: x.isin(VIOLENT_CATS).sum()),
        property_trials = ('offence_cat', lambda x: x.isin(PROPERTY_CATS).sum()),
        guilty_verdicts = ('verdict_cat', lambda x: x.isin(GUILTY_VERDICTS).sum()),
        not_guilty_verdicts = ('verdict_cat', lambda x: x.isin(NOT_GUILTY_VERDICTS).sum()),
        death_sentences = ('punishment_cat', lambda x: x.isin(DEATH_PUNISH).sum()),
        transport_sentences = ('punishment_cat', lambda x: x.isin(TRANSPORT_PUNISH).sum()),
        corporal_sentences = ('punishment_cat', lambda x: x.isin(CORPORAL_PUNISH).sum()),
    ).reset_index()

    def calculate_punishment_score(row):
        if row['punishment_cat'] in DEATH_PUNISH: return 5
        if row['punishment_cat'] in TRANSPORT_PUNISH: return 4
        if row['punishment_cat'] in CORPORAL_PUNISH: return 3
        return 0
    structured_df['punish_score'] = structured_df.apply(calculate_punishment_score, axis=1)
    convicted_df = structured_df[structured_df['verdict_cat'].isin(GUILTY_VERDICTS)]
    weekly_avg_punish_score = convicted_df.groupby('week_date')['punish_score'].mean().reset_index().rename(columns={'punish_score': 'avg_punishment_score'})
    weekly_metrics = pd.merge(weekly_counts, weekly_avg_punish_score, on='week_date', how='left')

    metric_cols_calc = ['violent_crime_prop', 'property_crime_prop', 'conviction_rate',
                        'death_sentence_rate', 'transport_rate', 'avg_punishment_score']
    for col in metric_cols_calc: weekly_metrics[col] = np.nan
    valid_week_mask = weekly_metrics['total_trials'] >= MIN_TRIALS_PER_WEEK
    logger.info(f"Calculating rates/proportions for {valid_week_mask.sum()} weeks with >= {MIN_TRIALS_PER_WEEK} trials.")
    total_trials_denom_valid = weekly_metrics.loc[valid_week_mask, 'total_trials']
    guilty_denom_valid = weekly_metrics.loc[valid_week_mask, 'guilty_verdicts'].replace(0, np.nan)
    valid_verdicts_denom_valid = (weekly_metrics.loc[valid_week_mask, 'guilty_verdicts'] + weekly_metrics.loc[valid_week_mask, 'not_guilty_verdicts']).replace(0, np.nan)
    weekly_metrics.loc[valid_week_mask, 'violent_crime_prop'] = weekly_metrics['violent_trials'] / total_trials_denom_valid
    weekly_metrics.loc[valid_week_mask, 'property_crime_prop'] = weekly_metrics['property_trials'] / total_trials_denom_valid
    weekly_metrics.loc[valid_week_mask, 'conviction_rate'] = weekly_metrics['guilty_verdicts'] / valid_verdicts_denom_valid
    weekly_metrics.loc[valid_week_mask, 'death_sentence_rate'] = weekly_metrics['death_sentences'] / guilty_denom_valid
    weekly_metrics.loc[valid_week_mask, 'transport_rate'] = weekly_metrics['transport_sentences'] / guilty_denom_valid
    weekly_metrics['avg_punishment_score'].fillna(0, inplace=True)

    logger.info("Imputing NaNs created by minimum trial filter (Structured)...")
    for col in metric_cols_calc:
        if weekly_metrics[col].isnull().any():
            weekly_metrics[col] = weekly_metrics[col].fillna(weekly_metrics[col].rolling(SMOOTHING_WINDOW*2, min_periods=1, center=True).median())
            weekly_metrics[col].fillna(weekly_metrics[col].median(), inplace=True)

    logger.info(f"Applying {SMOOTHING_WINDOW}-week rolling average smoothing (Structured)...")
    metric_cols_smooth = ['violent_crime_prop', 'property_crime_prop', 'conviction_rate',
                          'death_sentence_rate', 'transport_rate', 'avg_punishment_score']
    smoothed_structural_col_names = {}
    for col in metric_cols_smooth:
         smooth_col = f"{col}_smooth"
         smoothed_structural_col_names[col] = smooth_col
         weekly_metrics[smooth_col] = weekly_metrics[col].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
         weekly_metrics[smooth_col].fillna(weekly_metrics[col], inplace=True) # Fill NaNs from smoothing

    # --- Aggregate Sentiment Scores (Weekly Mean) ---
    if sentiment_cols_to_agg: # Check if there are sentiment columns to aggregate
        logger.info(f"Aggregating weekly sentiment scores (mean) for columns: {sentiment_cols_to_agg}...")
        weekly_sentiment_aggregated = sentiment_df.groupby('week_date')[sentiment_cols_to_agg].mean().reset_index()
        logger.info(f"Aggregated weekly sentiment scores for {weekly_sentiment_aggregated.shape[0]} weeks.")
    else:
        logger.info("Skipping sentiment aggregation as no sentiment columns were provided or calculated.")
        weekly_sentiment_aggregated = pd.DataFrame(columns=['week_date']) # Empty df with date col

    # --- Merge All Aggregated Data ---
    logger.info("Merging aggregated structured metrics and sentiment scores...")
    structured_to_merge = weekly_metrics[['week_date', 'total_trials'] + list(smoothed_structural_col_names.values())]
    # Merge structured with sentiment (use outer to keep all weeks)
    combined_weekly_metrics = pd.merge(structured_to_merge, weekly_sentiment_aggregated, on='week_date', how='outer')

    # --- Merge Combined Metrics with Mortality ---
    logger.info(f"Merging combined weekly metrics with weekly mortality data...")
    merged_df = pd.merge(weekly_mortality_df, combined_weekly_metrics, on='week_date', how='outer')
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)

    # --- Handle Missing Data (Post-Merge) & Imputation ---
    # Define all potential metric columns after merge
    all_metric_cols_post_merge = (['total_trials'] + list(smoothed_structural_col_names.values()) + sentiment_cols_to_agg)
    for col in all_metric_cols_post_merge:
        if col in merged_df.columns: # Check if column exists after merge
            missing_count = merged_df[col].isnull().sum()
            if missing_count > 0:
                logger.warning(f"{missing_count} weeks missing '{col}'. Imputing with rolling median then 0.")
                merged_df[col] = merged_df[col].fillna(merged_df[col].rolling(4, min_periods=1, center=True).median())
                merged_df[col].fillna(0, inplace=True) # Fallback fill with 0
        # No need to log warning if column doesn't exist (e.g., sentiment was skipped)

    missing_deaths = merged_df['deaths'].isnull().sum()
    if missing_deaths > 0: logger.warning(f"{missing_deaths} weeks missing mortality. Imputing 0."); merged_df['deaths'].fillna(0, inplace=True)
    merged_df['year'] = merged_df['year'].fillna(merged_df['week_date'].dt.isocalendar().year).astype(int)
    merged_df['week_of_year'] = merged_df['week_of_year'].fillna(merged_df['week_date'].dt.isocalendar().week).astype(int)

    # --- Log Transform Deaths ---
    merged_df['log_deaths'] = np.log1p(merged_df['deaths'])
    logger.info("Applied log1p transformation to 'deaths' column -> 'log_deaths'.")

    # --- Create Lagged Features ---
    # Choose columns to lag (examples: smoothed structured + aggregated sentiment)
    cols_to_lag = [
        smoothed_structural_col_names.get('conviction_rate'), # Use .get() for safety
        'hardship_sentiment' # Example sentiment
    ]
    cols_to_lag = [col for col in cols_to_lag if col is not None and col in merged_df.columns] # Filter out None or missing columns
    lagged_col_names = []
    for lag_col_name in cols_to_lag:
        feature_lag_col_name = f'{lag_col_name}_lag{LAG_WEEKS}w'
        lagged_col_names.append(feature_lag_col_name)
        logger.info(f"Creating lagged feature: '{feature_lag_col_name}' ({LAG_WEEKS} weeks)...")
        merged_df[feature_lag_col_name] = merged_df[lag_col_name].shift(LAG_WEEKS)
        initial_nan_count = merged_df[feature_lag_col_name].isnull().sum()
        if initial_nan_count > 0:
            merged_df[feature_lag_col_name] = merged_df[feature_lag_col_name].fillna(merged_df[feature_lag_col_name].rolling(4, min_periods=1, center=True).median())
            merged_df[feature_lag_col_name].fillna(0, inplace=True)

    # --- Standardize Features ---
    # Standardize smoothed structured, aggregated sentiment, and their lags
    features_to_standardize = list(smoothed_structural_col_names.values()) + sentiment_cols_to_agg + lagged_col_names
    standardized_col_names_map = {}
    logger.info(f"Attempting to standardize: {features_to_standardize}")
    for col in features_to_standardize:
        if col in merged_df.columns:
             std_col_name = f'{col}_std'
             standardized_col_names_map[col] = std_col_name
             logger.info(f"Standardizing '{col}' -> '{std_col_name}'.")
             scaler = StandardScaler()
             # Check for NaNs/Infs before scaling
             if merged_df[[col]].isnull().any().any() or np.isinf(merged_df[[col]]).any().any():
                 logger.warning(f"NaNs or Infs found in '{col}' before standardization. Imputing with median again.")
                 col_median = merged_df[col].median()
                 merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan).fillna(col_median)
             try:
                 merged_df[std_col_name] = scaler.fit_transform(merged_df[[col]])
                 merged_df[std_col_name] = merged_df[std_col_name].astype(float)
             except ValueError as e:
                 logger.error(f"StandardScaler failed for column '{col}': {e}. Skipping standardization for this column.")
                 # Remove from map if failed
                 if col in standardized_col_names_map: del standardized_col_names_map[col]
        else: logger.warning(f"Column '{col}' not found for standardization.")


    # --- Prepare Final DataFrame ---
    merged_df = merged_df.sort_values("week_date").reset_index(drop=True)
    merged_df["time_idx"] = (merged_df["week_date"] - merged_df["week_date"].min()).dt.days // 7

    # Ensure dtypes before final selection
    merged_df["log_deaths"] = merged_df["log_deaths"].astype(float)
    merged_df["deaths"] = merged_df["deaths"].astype(float)
    for col in list(smoothed_structural_col_names.values()) + sentiment_cols_to_agg + lagged_col_names:
        if col in merged_df.columns: merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').astype(float)
    merged_df["year"] = merged_df["year"].astype(str)
    merged_df["week_of_year"] = merged_df["week_of_year"].astype(str)
    merged_df["series_id"] = "London"

    # Define final columns list carefully
    final_cols = (
        ["week_date", "time_idx", "deaths", "log_deaths", "year", "week_of_year", "series_id", "total_trials"] +
        list(smoothed_structural_col_names.values()) + # Smoothed structured
        sentiment_cols_to_agg +                 # Aggregated sentiment
        lagged_col_names +                      # Lagged versions
        list(standardized_col_names_map.values()) # Standardized versions that were successfully created
    )
    # Ensure only unique columns that actually exist are selected
    final_cols_unique_exist = []
    for col in final_cols:
        if col in merged_df.columns and col not in final_cols_unique_exist:
            final_cols_unique_exist.append(col)

    merged_df = merged_df[final_cols_unique_exist].copy()

    logger.info(f"Shape before final NaN drop and clipping: {merged_df.shape}")
    logger.info(f"Columns before final NaN drop: {merged_df.columns.tolist()}")

    # Define critical columns for NaN drop using the map of standardized names
    critical_cols_for_model = ['time_idx', 'log_deaths'] + list(standardized_col_names_map.values())
    critical_cols_for_model_exist = [col for col in critical_cols_for_model if col in merged_df.columns]
    logger.info(f"Dropping NaNs based on columns: {critical_cols_for_model_exist}")
    merged_df.dropna(subset=critical_cols_for_model_exist, inplace=True)
    logger.info(f"Shape after final NaN drop: {merged_df.shape}")

    # --- Clip Ends ---
    if not merged_df.empty:
        weeks_to_clip = 52
        min_idx_df = merged_df['time_idx'].min()
        max_idx_df = merged_df['time_idx'].max()
        original_rows = len(merged_df)
        if max_idx_df >= min_idx_df and (max_idx_df - min_idx_df + 1) > 2 * weeks_to_clip :
            merged_df = merged_df[(merged_df['time_idx'] >= min_idx_df + weeks_to_clip) &
                                  (merged_df['time_idx'] <= max_idx_df - weeks_to_clip)].copy()
            logger.info(f"Clipped ends: Removed first/last {weeks_to_clip} weeks. New shape: {merged_df.shape}. ({original_rows - len(merged_df)} rows removed)")
        else: logger.warning(f"Not enough data span ({max_idx_df - min_idx_df + 1} weeks) to clip {weeks_to_clip} weeks. Skipping clipping.")

        if merged_df.empty:
            logger.error("DataFrame empty after clipping ends.")
            return pd.DataFrame(columns=final_cols_unique_exist)

        # Check sufficiency for TFT *after* clipping
        if merged_df["time_idx"].max() - merged_df["time_idx"].min() + 1 < WEEKLY_MAX_ENCODER_LENGTH + WEEKLY_MAX_PREDICTION_LENGTH:
             logger.error(f"Insufficient weekly data span ({merged_df.shape[0]} weeks after clipping) for TFT config.")
             return pd.DataFrame(columns=final_cols_unique_exist)
    else:
        logger.error("DataFrame is empty after NaN drop, cannot proceed.")
        return pd.DataFrame(columns=final_cols_unique_exist)

    logger.info(f"Final weekly combined metrics data shape returned: {merged_df.shape}. Time idx range: {merged_df['time_idx'].min()}-{merged_df['time_idx'].max()}")
    logger.info(f"Final Columns Returned: {merged_df.columns.tolist()}")
    nan_check_after = merged_df.isnull().sum()
    if nan_check_after.any(): logger.warning(f"NaNs still present after final processing:\n{nan_check_after[nan_check_after > 0]}")

    return merged_df

# -----------------------------------------------------------------------------
# 5. TFT Training and Evaluation (WEEKLY)
# -----------------------------------------------------------------------------
def train_tft_model(df: pd.DataFrame,
                    time_varying_reals_cols: List[str],
                    run_name: str, # <<< ADDED run_name for logging/checkpoints
                    # --- Keep other parameters ---
                    max_epochs: int = 75, # <<< Increased default max_epochs
                    batch_size: int = BATCH_SIZE,
                    encoder_length: int = WEEKLY_MAX_ENCODER_LENGTH,
                    pred_length: int = WEEKLY_MAX_PREDICTION_LENGTH,
                    lr: float = LEARNING_RATE,
                    hidden_size: int = HIDDEN_SIZE, # Keep complexity same for now
                    attn_heads: int = ATTENTION_HEAD_SIZE, # Keep complexity same for now
                    dropout: float = DROPOUT,
                    hidden_cont_size: int = HIDDEN_CONTINUOUS_SIZE, # Keep complexity same for now
                    clip_val: float = GRADIENT_CLIP_VAL) -> Tuple[Optional[TemporalFusionTransformer], Optional[pl.Trainer], Optional[torch.utils.data.DataLoader], Optional[TimeSeriesDataSet]]:
    """
    Trains the Temporal Fusion Transformer model on WEEKLY data using log-transformed deaths
    and dynamically specified real-valued features. Logs under a specific run name.
    """

    logger.info(f"--- Starting TFT Training for Run: '{run_name}' ---")
    logger.info(f"Setting up WEEKLY TFT model training (Target: log_deaths)...")
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
        required_numeric_cols = ["time_idx", "log_deaths", "deaths"] + time_varying_reals_cols
        required_numeric_cols = list(set(required_numeric_cols))
        for col in required_numeric_cols:
            if col not in data_for_tft.columns: raise ValueError(f"Column '{col}' not found.")
            data_for_tft[col] = pd.to_numeric(data_for_tft[col], errors='coerce')
        categorical_cols = ["series_id", "week_of_year", "year"]
        for col in categorical_cols: data_for_tft[col] = data_for_tft[col].astype(str)
        numeric_cols_to_impute = [col for col in required_numeric_cols if col != 'time_idx']
        if data_for_tft[numeric_cols_to_impute].isnull().any().any():
             nan_counts = data_for_tft[numeric_cols_to_impute].isnull().sum()
             logger.warning(f"NaNs found after casting:\n{nan_counts[nan_counts > 0]}. Imputing with median.")
             for col in numeric_cols_to_impute: data_for_tft[col].fillna(data_for_tft[col].median(), inplace=True)
        logger.info("Dtype check passed.")
    except Exception as e: logger.error(f"Error during dtype check: {e}", exc_info=True); return None, None, None, None

    # --- TimeSeriesDataSet Setup ---
    logger.info("Setting up WEEKLY TimeSeriesDataSet for TFT (Target: log_deaths)...")
    try:
        unknown_reals_for_tft = [col for col in time_varying_reals_cols if col != "log_deaths"]
        logger.info(f"Passing to TimeSeriesDataSet time_varying_unknown_reals: {unknown_reals_for_tft}")
        missing_tft_cols = [col for col in unknown_reals_for_tft if col not in data_for_tft.columns]
        if missing_tft_cols: raise ValueError(f"Columns for TFT `time_varying_unknown_reals` are missing: {missing_tft_cols}")

        training_dataset = TimeSeriesDataSet(
            data_for_tft[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx", target="log_deaths", group_ids=["series_id"],
            max_encoder_length=encoder_length, max_prediction_length=pred_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["week_of_year"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=["year"],
            time_varying_unknown_reals=unknown_reals_for_tft,
            add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=True,
            categorical_encoders={"year": NaNLabelEncoder(add_nan=True), "week_of_year": NaNLabelEncoder(add_nan=True)}
        )
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data_for_tft, predict=True, stop_randomization=True)

        effective_batch_size = max(1, min(batch_size, len(training_dataset) // 2 if len(training_dataset) > 1 else 1))
        # Ensure validation batch size isn't larger than the validation dataset size
        # Calculate size of validation part of data_for_tft
        val_data_len = len(data_for_tft[lambda x: x.time_idx > training_cutoff])
        # A simple estimate for validation samples (can be complex due to sequence overlaps)
        # A safer bet is often to use a smaller batch size for validation if unsure
        val_batch_size = max(1, min(effective_batch_size * 2, val_data_len // pred_length if pred_length > 0 else val_data_len))
        logger.info(f"Using effective train batch size: {effective_batch_size}, val batch size: {val_batch_size}")


        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=effective_batch_size, num_workers=0, persistent_workers=False)
        # Ensure shuffle=False for validation loader
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=val_batch_size, num_workers=0, persistent_workers=False, shuffle=False)

        if len(train_dataloader) == 0 or len(val_dataloader) == 0: logger.error("Empty dataloader(s)."); return None, None, None, None
    except Exception as e: logger.error(f"Error creating TimeSeriesDataSet/Dataloaders: {e}", exc_info=True); logger.error(f"Data info:\n{data_for_tft.info()}"); return None, None, None, None

    logger.info("Configuring TemporalFusionTransformer model...")
    loss_metric = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    try:
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset, learning_rate=lr, hidden_size=hidden_size, attention_head_size=attn_heads,
            dropout=dropout, hidden_continuous_size=hidden_cont_size, loss=loss_metric, log_interval=10, # Log slightly more often
            optimizer="adam", reduce_on_plateau_patience=5,
        )
        logger.info(f"TFT model parameters: {tft.size()/1e6:.1f} million")
    except Exception as e: logger.error(f"Error initializing TFT: {e}", exc_info=True); return None, None, val_dataloader, validation_dataset

    # --- Use run_name for logger ---
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=True, mode="min") # <<< Increased patience
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accelerator, devices = ('cpu', 1)
    logger.info(f"Configuring Trainer (Accelerator: {accelerator}, Devices: {devices})...")
    from pytorch_lightning.loggers import TensorBoardLogger
    # Use run_name to create distinct log directories
    tb_logger = TensorBoardLogger(save_dir="lightning_logs/", name=f"tft_{run_name}_weekly_log_target")
    trainer = pl.Trainer(
        max_epochs=max_epochs, # Use increased max_epochs from args
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=clip_val,
        callbacks=[lr_monitor, early_stop_callback],
        logger=tb_logger,
        enable_progress_bar=True
    )

    logger.info(f"Starting TFT model training for run '{run_name}'...")
    start_train_time = time.time()
    best_tft = None # Initialize best_tft
    try:
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        logger.info(f"TFT training finished for '{run_name}' in {(time.time() - start_train_time)/60:.2f} minutes.")
        best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback else None

        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model for '{run_name}' from checkpoint: {best_model_path}")
            # Load onto the globally defined DEVICE
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=DEVICE)
            logger.info(f"Best model loaded successfully for run '{run_name}'.")
            return best_tft, trainer, val_dataloader, validation_dataset
        else:
            logger.warning(f"Best checkpoint not found or invalid path for '{run_name}'. Attempting to return last model state.")
            # Try to get the last model state from the trainer
            last_model = trainer.model if hasattr(trainer, 'model') and trainer.model is not None else tft
            if last_model:
                 last_model.to(DEVICE) # Ensure it's on the correct device
                 logger.info(f"Returning last model state for run '{run_name}'.")
                 return last_model, trainer, val_dataloader, validation_dataset
            else:
                 logger.error(f"Could not retrieve last model state for run '{run_name}'.")
                 return None, trainer, val_dataloader, validation_dataset

    except Exception as e:
        logger.error(f"Error during TFT fitting for run '{run_name}': {e}", exc_info=True)
        # Attempt to return last model state even on error
        last_model = trainer.model if hasattr(trainer, 'model') and trainer.model is not None else tft
        if last_model: last_model.to(DEVICE)
        return last_model, trainer, val_dataloader, validation_dataset

# --- evaluate_model function (Correct for Forecasting Task) ---
# Ensure these imports are present at the top of your script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss # Or specific metric classes used
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import torch

# Make sure DEVICE is defined globally (e.g., DEVICE = torch.device("cpu"))
# Make sure logger is defined globally (e.g., logger = logging.getLogger(__name__))

# --- evaluate_model function (Correct for Forecasting Task - Revised Unpacking) ---
def evaluate_model(model: TemporalFusionTransformer, dataloader: torch.utils.data.DataLoader,
                   dataset: TimeSeriesDataSet, plot_dir: str, run_name: str) -> Dict[str, float]:
    """
    Evaluates TFT model on log_deaths, returns metrics (MAE, MSE, SMAPE) on original death scale,
    saves plots with run_name prefix, and acknowledges confidence-based metrics.
    Includes revised predict() output handling.
    """
    logger.info(f"Evaluating model performance for run '{run_name}'...")
    results = {}
    if model is None or dataloader is None or len(dataloader) == 0:
        logger.error("Model/Dataloader missing for evaluation.")
        return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    try:
        eval_device = next(model.parameters()).device
    except Exception:
        eval_device = torch.device(DEVICE) # Fallback to global DEVICE
        try:
            model.to(eval_device)
        except Exception as device_err:
            logger.error(f"Could not move model to device {eval_device}: {device_err}")
            eval_device = torch.device("cpu") # Force CPU if move fails
            model.to(eval_device)
            logger.warning("Forcing evaluation on CPU.")
    logger.info(f"Evaluation device: {eval_device}")

    actuals_log_list, preds_log_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(dataloader)):
            try:
                x_gpu = {k: v.to(eval_device) for k, v in x.items() if isinstance(v, torch.Tensor)}
                target_log = y[0].to(eval_device)
                preds = model(x_gpu)["prediction"] # Get predictions from model forward pass
                preds_log_list.append(preds.cpu())
                actuals_log_list.append(target_log.cpu())
            except Exception as batch_err:
                 logger.error(f"Error processing evaluation batch {i}: {batch_err}", exc_info=True)
                 continue # Skip problematic batch

    if not preds_log_list:
        logger.error("No predictions collected during evaluation.")
        return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}

    # --- Metric Calculation ---
    try:
        actuals_log_all = torch.cat(actuals_log_list).numpy()
        preds_log_all = torch.cat(preds_log_list).numpy()

        actuals_log_flat = actuals_log_all.flatten()
        # Ensure preds_log_all has the expected 3 dimensions (batch, time, quantiles)
        if preds_log_all.ndim != 3 or preds_log_all.shape[2] != 3:
             logger.error(f"Prediction tensor has unexpected shape: {preds_log_all.shape}. Expected (..., 3). Cannot calculate metrics.")
             return {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan}
        preds_log_median_flat = preds_log_all[:, :, 1].flatten() # Use median (p50) - index 1

        min_len_m = min(len(actuals_log_flat), len(preds_log_median_flat))
        if len(actuals_log_flat) != len(preds_log_median_flat):
            logger.warning(f"Metric length mismatch ({len(actuals_log_flat)} vs {len(preds_log_median_flat)}): Truncating.")
            preds_log_median_flat = preds_log_median_flat[:min_len_m]
            actuals_log_flat = actuals_log_flat[:min_len_m]

        # Inverse transform to original scale
        actuals_orig_flat = np.expm1(actuals_log_flat)
        preds_orig_median_flat = np.maximum(0, np.expm1(preds_log_median_flat)) # Ensure non-negative

        val_mae = mean_absolute_error(actuals_orig_flat, preds_orig_median_flat)
        val_mse = mean_squared_error(actuals_orig_flat, preds_orig_median_flat)

        # Calculate SMAPE carefully, avoiding division by zero
        denominator = (np.abs(actuals_orig_flat) + np.abs(preds_orig_median_flat)) / 2.0
        # Handle cases where both actual and prediction are near zero
        smape_mask = denominator > 1e-9 # Use a small threshold instead of exact zero
        val_smape = np.mean(
            np.abs(preds_orig_median_flat[smape_mask] - actuals_orig_flat[smape_mask]) /
            denominator[smape_mask]
            ) * 100 if np.any(smape_mask) else 0.0 # Return 0 if all denominators are zero

        results = {"MAE": val_mae, "MSE": val_mse, "SMAPE": val_smape}
        logger.info(f"[Validation Metrics ({run_name}, Original Scale)] MAE={val_mae:.3f} MSE={val_mse:.3f} SMAPE={val_smape:.3f}%")
        logger.info("Note: While standard forecasting metrics (MAE, MSE, SMAPE) are reported, evaluating nuanced historical sentiment ideally involves confidence-based metrics like cPrecision/cRecall (Yacouby & Axman, 2020), which were beyond the scope of direct implementation for this forecasting task.")

    except Exception as metric_err:
        logger.error(f"Error calculating evaluation metrics: {metric_err}", exc_info=True)
        results = {"MAE": np.nan, "MSE": np.nan, "SMAPE": np.nan} # Ensure results dict exists

    logger.info(f"Generating weekly evaluation plots for run '{run_name}' (showing original death scale)...")
    plot_fig = None; plot_fig_res = None
    try:
        # --- Simplest predict call for plotting data ---
        # This should return quantiles by default, along with x and index
        logger.info("Calling model.predict() for plotting...")
        predictions = model.predict(dataloader, return_x=True, return_index=True)
        logger.info(f"model.predict() output type: {type(predictions)}")
        # --- End Simplification ---

        # --- Direct Unpacking Attempt ---
        if not (isinstance(predictions, (list, tuple)) and len(predictions) == 3):
             # Check if it's a dict (less common direct return)
             if isinstance(predictions, dict) and 'prediction' in predictions and 'x' in predictions and 'index' in predictions:
                 logger.warning("Predict returned a dict, unpacking components.")
                 raw_preds_container = predictions
                 x_output = predictions['x']
                 index_df = predictions['index']
                 # Need to extract the prediction tensor itself if mode was raw/prediction
                 if 'prediction' in raw_preds_container:
                     raw_preds = raw_preds_container['prediction']
                 else:
                     logger.error("Prediction tensor missing in returned dict. Skipping plots.")
                     return results
             else:
                 logger.error(f"Predict output did not return expected tuple/list of 3 or usable dict. Got {type(predictions)}. Skipping plots.")
                 return results # Return metrics calculated earlier
        else:
            # Standard tuple/list unpacking
            raw_preds, x_output, index_df = predictions # Unpack directly

        # Validate components (more detailed checks)
        # Check raw_preds: should be tensor [samples, time, quantiles] for mode=quantiles (default)
        if not isinstance(raw_preds, torch.Tensor) or raw_preds.ndim != 3 or raw_preds.shape[2] != 3:
            logger.error(f"Unpacked prediction component is not a valid quantile tensor. Shape: {raw_preds.shape if isinstance(raw_preds, torch.Tensor) else type(raw_preds)}. Skipping plots.")
            return results
        # Check x_output
        if not isinstance(x_output, dict) or 'decoder_target' not in x_output:
             logger.error(f"Unpacked x_output is not a dict or missing 'decoder_target'. Type: {type(x_output)}. Skipping plots.")
             return results
        # Check index_df
        if not isinstance(index_df, pd.DataFrame) or 'time_idx' not in index_df.columns:
             logger.error(f"Unpacked index_df is not a DataFrame or missing 'time_idx'. Type: {type(index_df)}. Skipping plots.")
             return results
        # --- End Validation ---

        # Proceed with plotting
        preds_log_tensor = raw_preds.cpu()
        actuals_log_tensor = x_output['decoder_target'].cpu()
        time_idx_flat = index_df['time_idx'].values

        # Flatten log predictions (p10, p50, p90)
        preds_log_p10_flat = preds_log_tensor[:, :, 0].flatten().numpy()
        preds_log_p50_flat = preds_log_tensor[:, :, 1].flatten().numpy()
        preds_log_p90_flat = preds_log_tensor[:, :, 2].flatten().numpy()
        actuals_log_flat_plot = actuals_log_tensor.flatten().numpy()

        # Check lengths and truncate if necessary
        n_preds = len(preds_log_p50_flat); n_actuals = len(actuals_log_flat_plot); n_time = len(time_idx_flat)
        if not (n_preds == n_actuals == n_time):
            min_len_plot = min(n_preds, n_actuals, n_time)
            if min_len_plot == 0: logger.error("Zero length plot data."); return results
            logger.warning(f"Plot length mismatch ({n_preds} vs {n_actuals} vs {n_time}): Truncating.")
            preds_log_p10_flat=preds_log_p10_flat[:min_len_plot]; preds_log_p50_flat=preds_log_p50_flat[:min_len_plot]; preds_log_p90_flat=preds_log_p90_flat[:min_len_plot]
            actuals_log_flat_plot=actuals_log_flat_plot[:min_len_plot]; time_idx_flat=time_idx_flat[:min_len_plot]

        # Inverse transform ACTUALS and PREDICTIONS for plotting
        actuals_orig_flat_plot = np.expm1(actuals_log_flat_plot)
        p10_orig_flat = np.maximum(0, np.expm1(preds_log_p10_flat))
        p50_orig_flat = np.maximum(0, np.expm1(preds_log_p50_flat))
        p90_orig_flat = np.maximum(0, np.expm1(preds_log_p90_flat))

        # Sort by time_idx for plotting
        sort_indices = np.argsort(time_idx_flat)
        time_idx_sorted=time_idx_flat[sort_indices]; actuals_sorted=actuals_orig_flat_plot[sort_indices]
        p10_sorted=p10_orig_flat[sort_indices]; p50_sorted=p50_orig_flat[sort_indices]; p90_sorted=p90_orig_flat[sort_indices]

        # --- Generate Forecast Plot ---
        plot_fig, ax = plt.subplots(figsize=(18, 7)) # Wider plot
        ax.plot(time_idx_sorted, actuals_sorted, label="Actual Deaths", marker='.', linestyle='-', alpha=0.7, color='black', markersize=3, linewidth=0.8)
        ax.plot(time_idx_sorted, p50_sorted, label="Predicted Median (p50)", linestyle='--', alpha=0.9, color='tab:orange', linewidth=1.2)
        ax.fill_between(time_idx_sorted, p10_sorted, p90_sorted, color='tab:orange', alpha=0.3, label='p10-p90 Quantiles')
        plot_title = f"TFT Forecast vs Actuals ({run_name})\nMAE={val_mae:.2f}, SMAPE={val_smape:.2f}%"
        ax.set_title(plot_title, fontsize=14)
        try:
             # Attempt to get start date from dataset's raw data for better label
             start_date_dt = pd.to_datetime(dataset.data["raw"]["week_date"].iloc[time_idx_sorted[0]])
             x_label = f"Time Index (Weeks since {start_date_dt:%Y-%m-%d})"
        except Exception:
             x_label = "Time Index (Weeks)" # Fallback label
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Weekly Deaths (Original Scale)", fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        plot_file = os.path.join(plot_dir, f"tft_val_forecast_{run_name}_original_scale.png")
        plt.savefig(plot_file); logger.info(f"Saved forecast plot for '{run_name}' to {plot_file}");
        plt.close(plot_fig) # Close this specific figure

        # --- Residual Plot ---
        residuals = actuals_sorted - p50_sorted
        plot_fig_res, ax_res = plt.subplots(figsize=(10, 6))
        ax_res.scatter(p50_sorted, residuals, alpha=0.3, s=15, color='tab:blue', edgecolors='k', linewidth=0.5)
        ax_res.axhline(0, color='red', linestyle='--', linewidth=1)
        ax_res.set_title(f'Residual Plot ({run_name}, Original Scale)', fontsize=14)
        ax_res.set_xlabel('Predicted Median Deaths (Original Scale)', fontsize=12)
        ax_res.set_ylabel('Residuals (Original Scale)', fontsize=12)
        ax_res.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        save_path_res = os.path.join(plot_dir, f"residuals_{run_name}_original_scale_plot.png")
        plt.savefig(save_path_res); logger.info(f"Saved residual plot for '{run_name}' to {save_path_res}");
        plt.close(plot_fig_res) # Close this specific figure

    except Exception as e:
        logger.warning(f"Evaluation plotting failed: {e}", exc_info=True)
    finally:
        # Attempt to close any figures that might still be open
        if 'plot_fig' in locals() and plot_fig is not None and plt.fignum_exists(plot_fig.number): plt.close(plot_fig)
        if 'plot_fig_res' in locals() and plot_fig_res is not None and plt.fignum_exists(plot_fig_res.number): plt.close(plot_fig_res)
        plt.close('all') # General cleanup

    return results

# -----------------------------------------------------------------------------
# 6. Enhanced Plotting Functions (UPDATED FOR WEEKLY)
# -----------------------------------------------------------------------------
# (Make sure plt is imported: import matplotlib.pyplot as plt)

def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str,
                     ylabel: str, filename: str, plot_dir: str,
                     zoom_ylim: Optional[Tuple[float, float]] = None,
                     mark_zero_threshold: float = 0.01):
    """
    Plots a simple time series, optionally zooming the y-axis and marking near-zero points.

    Args:
        df: DataFrame containing the data.
        time_col: Name of the column for the x-axis (e.g., 'week_date').
        value_col: Name of the column for the y-axis.
        title: Plot title.
        ylabel: Y-axis label.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
        zoom_ylim: Optional tuple (min_y, max_y) to set y-axis limits.
        mark_zero_threshold: If zoom_ylim is set, mark points below this threshold.
    """
    logger.info(f"Generating plot: {title}")
    if time_col not in df.columns or value_col not in df.columns:
        logger.error(f"Plot fail '{filename}': Missing columns '{time_col}' or '{value_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for plot '{filename}'. Skipping.")
        return

    try:
        fig, ax = plt.subplots(figsize=(18, 6)) # Wider plot for weekly data

        # Plot the main series
        ax.plot(df[time_col], df[value_col], marker='.', linestyle='-',
                markersize=1.5, alpha=0.6, linewidth=0.8, label=ylabel)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(time_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Handle y-axis zoom and zero marking
        if zoom_ylim is not None:
            try:
                min_y, max_y = zoom_ylim
                # Find points below the threshold
                zero_mask = df[value_col] < mark_zero_threshold
                if zero_mask.any():
                    # Plot markers for zero/near-zero points just below the zoomed area
                    # Adjust marker_y_pos if min_y can be negative
                    marker_y_pos = min_y - (max_y - min_y) * 0.02 # Position slightly below min_y
                    ax.plot(df.loc[zero_mask, time_col],
                            [marker_y_pos] * zero_mask.sum(),
                            marker='|', linestyle='None', markersize=5, color='red', alpha=0.5,
                            label=f'< {mark_zero_threshold:.2f} threshold')
                    ax.legend() # Show legend including the threshold markers

                logger.info(f"Applying y-axis zoom: {zoom_ylim}")
                ax.set_ylim(zoom_ylim)
            except Exception as e:
                logger.error(f"Failed to apply zoom/mark zeros for '{filename}': {e}")
                # Fallback to default y-limits if zoom fails
                pass # Let matplotlib auto-scale

        plt.tight_layout()
        # Add rotation for dates if time_col is week_date
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
             fig.autofmt_xdate(rotation=45, ha='right')

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"Plot fail '{filename}': {e}", exc_info=True)
    finally:
        # Ensure figure is closed
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        else:
            plt.close() # Close the current figure implicitly created

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


def plot_sentiment_with_rolling_stats(df: pd.DataFrame, time_col: str, value_col: str,
                                      window: int, title: str, ylabel: str,
                                      filename: str, plot_dir: str):
    """
    Plots a sentiment time series along with its rolling mean and rolling standard deviation.

    Args:
        df: DataFrame containing the data.
        time_col: Name of the column for the x-axis (e.g., 'week_date').
        value_col: Name of the sentiment column for the y-axis.
        window: Integer window size for rolling statistics.
        title: Plot title.
        ylabel: Y-axis label for the raw score.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
    """
    logger.info(f"Generating rolling stats plot: {title}")
    if time_col not in df.columns or value_col not in df.columns:
        logger.error(f"Plot fail '{filename}': Missing columns '{time_col}' or '{value_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for plot '{filename}'. Skipping.")
        return

    try:
        # Calculate rolling statistics
        rolling_mean = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = df[value_col].rolling(window=window, center=True, min_periods=1).std()

        fig, ax1 = plt.subplots(figsize=(18, 7)) # Slightly taller for dual axis

        color1 = 'lightblue'
        ax1.set_xlabel(time_col.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel(ylabel, color=color1, fontsize=12)
        # Plot raw score lightly in the background
        ax1.plot(df[time_col], df[value_col], color=color1, label=f'{ylabel} (Raw)', alpha=0.4, linewidth=0.5, marker='.', markersize=1)
        # Plot rolling mean prominently
        line1 = ax1.plot(df[time_col], rolling_mean, color='blue', label=f'{ylabel} ({window}w Rolling Mean)', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        y_min, y_max = rolling_mean.min(), rolling_mean.max()
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range) # Auto-adjust ylim based on mean

        # Create a second y-axis for rolling std dev
        ax2 = ax1.twinx()
        color2 = 'darkorange'
        ax2.set_ylabel(f'{window}w Rolling Std Dev', color=color2, fontsize=12)
        line2 = ax2.plot(df[time_col], rolling_std, color=color2, label=f'{ylabel} ({window}w Rolling Std Dev)', linestyle='--', linewidth=1.2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0) # Std dev cannot be negative

        fig.suptitle(title, fontsize=14)
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout considering suptitle

        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
             fig.autofmt_xdate(rotation=45, ha='right')

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved rolling stats plot: {save_path}")

    except Exception as e:
        logger.error(f"Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        else:
            plt.close()

def plot_training_history(log_dir: str, run_name: str, plot_dir: str):
    """
    Plots training and validation loss curves from TensorBoard metrics.csv.

    Args:
        log_dir: Path to the specific version directory inside lightning_logs
                 (e.g., 'lightning_logs/my_run_name/version_0').
        run_name: Name of the training run (for plot title/filename).
        plot_dir: Directory to save the plot.
    """
    metrics_path = os.path.join(log_dir, "metrics.csv")
    logger.info(f"Attempting to plot training history for '{run_name}' from: {metrics_path}")

    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found at {metrics_path}. Skipping training history plot.")
        return

    plot_fig_train = None # Initialize figure variable
    try:
        metrics_df = pd.read_csv(metrics_path)

        # Filter for epoch-level metrics if step metrics are also present
        # Plot 'val_loss' and 'train_loss_epoch' if available
        epochs = metrics_df[metrics_df["val_loss"].notna()]["epoch"]
        val_loss = metrics_df[metrics_df["val_loss"].notna()]["val_loss"]

        # Check if train_loss_epoch exists, otherwise try train_loss_step (less ideal)
        if "train_loss_epoch" in metrics_df.columns:
            train_loss = metrics_df[metrics_df["train_loss_epoch"].notna()]["train_loss_epoch"]
            train_epochs = metrics_df[metrics_df["train_loss_epoch"].notna()]["epoch"]
            train_loss_label = "Train Loss (Epoch)"
        elif "train_loss_step" in metrics_df.columns:
             # Aggregate step loss by epoch (simple mean)
             logger.warning("train_loss_epoch not found, using mean train_loss_step per epoch.")
             train_loss_agg = metrics_df.dropna(subset=["train_loss_step"]).groupby("epoch")["train_loss_step"].mean()
             train_loss = train_loss_agg.values
             train_epochs = train_loss_agg.index
             train_loss_label = "Train Loss (Step Avg)"
        else:
            logger.warning("No training loss found in metrics.csv. Plotting validation loss only.")
            train_loss = None

        if val_loss.empty and (train_loss is None or len(train_loss) == 0) :
            logger.warning("No valid loss data found in metrics file. Skipping plot.")
            return

        plot_fig_train, ax = plt.subplots(figsize=(12, 6))

        if not val_loss.empty:
            ax.plot(epochs, val_loss, label="Validation Loss", marker='o', linestyle='-')
        if train_loss is not None and len(train_loss) > 0:
             # Ensure train_epochs aligns if using aggregated step loss
             if len(train_epochs) == len(train_loss):
                 ax.plot(train_epochs, train_loss, label=train_loss_label, marker='x', linestyle='--')
             else:
                 logger.warning("Length mismatch between train_epochs and train_loss. Skipping train loss plot.")


        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Quantile Loss)")
        ax.set_title(f"Training History - {run_name}")
        ax.legend()
        ax.grid(True, linestyle=':')
        plt.tight_layout()

        plot_filename = f"training_history_{run_name}.png"
        save_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(save_path)
        logger.info(f"Saved training history plot for '{run_name}' to {save_path}")

    except FileNotFoundError:
         logger.warning(f"Metrics file not found at {metrics_path}. Cannot plot training history.")
    except KeyError as e:
         logger.warning(f"Could not find expected columns ('val_loss', 'epoch', etc.) in {metrics_path}: {e}. Skipping plot.")
    except Exception as e:
        logger.error(f"Failed to plot training history for '{run_name}': {e}", exc_info=True)
    finally:
        if plot_fig_train is not None and plt.fignum_exists(plot_fig_train.number):
            plt.close(plot_fig_train)
        plt.close('all') # Close any other figures

def plot_ccf(df: pd.DataFrame, var1: str, var2: str, max_lags: int,
    title: str, filename: str, plot_dir: str):
    """Plots the Cross-Correlation Function (CCF) between two variables."""
    logger.info(f"Generating CCF plot: {title}")
    if var1 not in df.columns or var2 not in df.columns:
        logger.error(f"CCF Plot fail '{filename}': Missing columns '{var1}' or '{var2}'.")
        return
    if df[[var1, var2]].isnull().any().any():
        logger.warning(f"NaNs found in columns for CCF '{filename}'. Dropping NaNs for calculation.")
        df_ccf = df[[var1, var2]].dropna()
    else:
        df_ccf = df[[var1, var2]]

    if len(df_ccf) < max_lags * 2:
        logger.warning(f"Not enough data points ({len(df_ccf)}) for CCF plot '{filename}' with max_lags={max_lags}. Skipping.")
        return

    ccf_fig = None # Initialize figure variable
    try:
        # Calculate CCF - Note: ccf(x, y) calculates corr(x_{t+k}, y_t)
        # We want corr(var1_{t+k}, var2_t) -> var1 leads var2 for positive k
        # Or corr(var1_t, var2_{t+k}) -> var2 leads var1 for positive k
        # Let's calculate corr(var1_{t+k}, var2_t) -> var1 is leading for positive k
        correlation = ccf(df_ccf[var1], df_ccf[var2], adjusted=False) # Calculate raw correlation

        nlags = max_lags
        lags = np.arange(-nlags, nlags + 1)
        # Extract relevant lags from ccf output (it calculates for positive lags only relative to first series)
        # We need to reconstruct for negative lags as well
        ccf_vals = np.zeros(len(lags))
        # Positive lags (k >= 0): corr(var1_{t+k}, var2_t)
        ccf_vals[nlags:] = correlation[:nlags+1]
        # Negative lags (k < 0): corr(var1_{t-|k|}, var2_t) = corr(var1_t, var2_{t+|k|})
        # This requires calculating ccf in the other direction
        correlation_rev = ccf(df_ccf[var2], df_ccf[var1], adjusted=False)
        ccf_vals[:nlags] = correlation_rev[1:nlags+1][::-1] # Get lags 1 to nlags and reverse

        # Calculate confidence intervals (approximate for large samples)
        conf_level = 1.96 / np.sqrt(len(df_ccf))

        ccf_fig, ax = plt.subplots(figsize=(12, 5))
        # Use stem plot for correlations
        markerline, stemlines, baseline = ax.stem(lags, ccf_vals, linefmt='grey', markerfmt='o', basefmt='black')
        plt.setp(markerline, 'color', 'blue', 'markersize', 4)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 0.5)

        # Plot confidence intervals
        ax.axhline(conf_level, color='grey', linestyle='--', linewidth=0.8)
        ax.axhline(-conf_level, color='grey', linestyle='--', linewidth=0.8)
        ax.fill_between(lags, -conf_level, conf_level, color='grey', alpha=0.15)

        ax.set_xlabel(f"Lag (Weeks) - Positive lag means '{var1}' leads '{var2}'")
        ax.set_ylabel("Cross-correlation")
        ax.set_title(title)
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path); logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"CCF Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if ccf_fig is not None and plt.fignum_exists(ccf_fig.number):
            plt.close(ccf_fig)
        plt.close('all')


def plot_rolling_correlation(df: pd.DataFrame, time_col: str, var1: str, var2: str,
                           window: int, title: str, filename: str, plot_dir: str):
    """Plots the rolling correlation between two variables."""
    logger.info(f"Generating rolling correlation plot: {title}")
    if time_col not in df.columns or var1 not in df.columns or var2 not in df.columns:
        logger.error(f"Rolling Correlation fail '{filename}': Missing columns.")
        return
    if df[[var1, var2]].isnull().any().any():
        logger.warning(f"NaNs found in columns for rolling correlation '{filename}'. Calculation might produce NaNs.")

    roll_fig = None
    try:
        rolling_corr = df[var1].rolling(window=window, center=True, min_periods=window // 2).corr(df[var2])

        roll_fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(df[time_col], rolling_corr, label=f'{window//52}y Rolling Correlation', linewidth=1.5)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel(time_col.replace('_', ' ').title())
        ax.set_ylabel("Correlation Coefficient")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            roll_fig.autofmt_xdate(rotation=45, ha='right')
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path); logger.info(f"Saved plot: {save_path}")

    except Exception as e:
        logger.error(f"Rolling Correlation fail '{filename}': {e}", exc_info=True)
    finally:
        if roll_fig is not None and plt.fignum_exists(roll_fig.number):
            plt.close(roll_fig)
        plt.close('all')

def plot_acf_pacf(series: pd.Series, lags: int, title_prefix: str, filename_suffix: str, plot_dir: str):
    """Plots ACF and PACF for a given time series."""
    logger.info(f"Generating ACF/PACF plots for: {title_prefix}")
    if series.isnull().any():
        logger.warning(f"NaNs found in series for ACF/PACF '{title_prefix}'. Dropping NaNs.")
        series = series.dropna()
    if len(series) < lags * 2:
         logger.warning(f"Not enough data points ({len(series)}) for ACF/PACF plot '{title_prefix}' with lags={lags}. Skipping.")
         return

    acf_fig = None; pacf_fig = None
    try:
        # ACF Plot
        acf_fig = plt.figure(figsize=(12, 5))
        plot_acf(series, lags=lags, ax=acf_fig.gca(), title=f'{title_prefix} - Autocorrelation (ACF)')
        plt.tight_layout()
        acf_filename = f"acf_{filename_suffix}.png"
        plt.savefig(os.path.join(plot_dir, acf_filename)); logger.info(f"Saved plot: {acf_filename}")
        plt.close(acf_fig)

        # PACF Plot
        pacf_fig = plt.figure(figsize=(12, 5))
        # Method 'ywm' is often more stable for PACF
        plot_pacf(series, lags=lags, ax=pacf_fig.gca(), title=f'{title_prefix} - Partial Autocorrelation (PACF)', method='ywm')
        plt.tight_layout()
        pacf_filename = f"pacf_{filename_suffix}.png"
        plt.savefig(os.path.join(plot_dir, pacf_filename)); logger.info(f"Saved plot: {pacf_filename}")
        plt.close(pacf_fig)

    except Exception as e:
        logger.error(f"ACF/PACF Plot fail for '{title_prefix}': {e}", exc_info=True)
    finally:
         if acf_fig is not None and plt.fignum_exists(acf_fig.number): plt.close(acf_fig)
         if pacf_fig is not None and plt.fignum_exists(pacf_fig.number): plt.close(pacf_fig)
         plt.close('all')

def plot_lag_scatter(df: pd.DataFrame, target_col: str, predictor_col: str, lag: int,
                       title: str, filename: str, plot_dir: str):
    """
    Plots a scatter plot of the target variable vs. a lagged predictor variable.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column (e.g., 'log_deaths').
        predictor_col: Name of the predictor column (e.g., 'hardship_sentiment').
        lag: The number of time steps (weeks) to lag the predictor.
        title: Plot title.
        filename: Name for the saved plot file.
        plot_dir: Directory to save the plot.
    """
    logger.info(f"Generating Lag Scatter plot: {title} (Lag={lag})")
    if target_col not in df.columns or predictor_col not in df.columns:
        logger.error(f"Lag Scatter fail '{filename}': Missing columns '{target_col}' or '{predictor_col}'.")
        return
    if df.empty:
        logger.warning(f"DataFrame empty for lag scatter plot '{filename}'. Skipping.")
        return
    if lag <= 0:
        logger.error(f"Lag must be positive for lag scatter plot. Got {lag}. Skipping '{filename}'.")
        return

    lag_scatter_fig = None # Initialize figure variable
    try:
        # Create a temporary DataFrame with the lagged predictor
        df_lagged = df[[target_col, predictor_col]].copy()
        lagged_predictor_col_name = f"{predictor_col}_lag{lag}"
        df_lagged[lagged_predictor_col_name] = df_lagged[predictor_col].shift(lag)

        # Drop NaNs introduced by shifting
        df_lagged.dropna(subset=[target_col, lagged_predictor_col_name], inplace=True)

        if df_lagged.empty:
             logger.warning(f"No data remaining after creating lag={lag} for '{predictor_col}'. Skipping scatter plot.")
             return

        # Create the scatter plot
        lag_scatter_fig = plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_lagged, x=lagged_predictor_col_name, y=target_col,
                        alpha=0.2, s=10, edgecolor=None) # Same styling as previous scatter

        # Calculate correlation for the title
        try:
             # Use numpy for correlation after dropping NaNs
             correlation = np.corrcoef(df_lagged[target_col], df_lagged[lagged_predictor_col_name])[0, 1]
             plot_title = f"{title}\n({target_col} vs {predictor_col} at Lag {lag}w)\n(Correlation: {correlation:.2f})"
        except Exception as corr_err:
             logger.warning(f"Could not calculate correlation for lag scatter: {corr_err}")
             plot_title = f"{title}\n({target_col} vs {predictor_col} at Lag {lag}w)"


        plt.title(plot_title, fontsize=12) # Smaller fontsize maybe
        plt.xlabel(f"{predictor_col.replace('_', ' ').title()} (Lag {lag}w)")
        plt.ylabel(target_col.replace('_', ' ').title())
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        logger.info(f"Saved lag scatter plot: {save_path}")

    except Exception as e:
        logger.error(f"Lag Scatter Plot fail '{filename}': {e}", exc_info=True)
    finally:
        if lag_scatter_fig is not None and plt.fignum_exists(lag_scatter_fig.number):
            plt.close(lag_scatter_fig)
        plt.close('all')

# -----------------------------------------------------------------------------
# 7. Interpretation & Granger Causality (Unchanged Functions)
# -----------------------------------------------------------------------------
def interpret_tft(model: TemporalFusionTransformer, val_dataloader: torch.utils.data.DataLoader,
                  plot_dir: str, run_name: str): # Added run_name
    """
    Calculates and saves TFT feature importance plots with run_name prefix.
    Includes revised predict() output handling for interpretation.
    """
    logger.info(f"Calculating TFT feature importance for run '{run_name}'...")
    if model is None or val_dataloader is None or len(val_dataloader) == 0:
        logger.warning("Model/Dataloader missing, skip interpretation.")
        return

    matplotlib_imported = False; fig_imp = None
    try: import matplotlib.pyplot as plt; matplotlib_imported = True
    except ImportError: logger.warning("matplotlib not found. Skipping plot generation.")

    try:
        interpret_device = "cpu"; model.to(interpret_device)
        logger.info(f"Running interpretation on: {interpret_device}")

        logger.info("Calling model.predict(mode='raw') for interpretation...")
        raw_predictions_dict = model.predict(val_dataloader, mode="raw")
        logger.info(f"predict(mode='raw') output type: {type(raw_predictions_dict)}")
        # --- End Predict call ---

        # --- Validate output ---
        if not isinstance(raw_predictions_dict, dict) or 'prediction' not in raw_predictions_dict:
            logger.error(f"Predict(mode='raw') did not return expected dictionary. Got: {type(raw_predictions_dict)}. Skip interpretation.")
            return
        # --- End Validation ---

        # Move tensors to the correct device
        raw_predictions_dict_cpu = {k: v.to(interpret_device) if isinstance(v, torch.Tensor) else v
                                    for k, v in raw_predictions_dict.items()}

        # --- Calculate and Plot Interpretation ---
        if matplotlib_imported:
            interpretation = model.interpret_output(raw_predictions_dict, reduction="mean")
            logger.info("Plotting TFT interpretation...")

            fig_imp = model.plot_interpretation(interpretation, plot_type="variable_importance")
            fig_imp.suptitle(f"TFT Feature Importance ({run_name})") # Add run_name
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_imp = os.path.join(plot_dir, f"tft_interpretation_importance_{run_name}.png") # Use run_name
            fig_imp.savefig(save_path_imp)
            logger.info(f"Saved interpretation plot for '{run_name}' to {save_path_imp}")
            # Closing handled in finally
        else:
            logger.warning("Skipping interpretation plot generation as matplotlib could not be imported.")

    except AttributeError as e: logger.error(f"AttributeError during interpretation: {e}.", exc_info=True)
    except Exception as e: logger.error(f"Error during interpretation: {e}", exc_info=True)
    finally:
        if matplotlib_imported:
            if fig_imp is not None and plt.fignum_exists(fig_imp.number): plt.close(fig_imp)
            plt.close('all')
        model.to(DEVICE)

# Make sure InfeasibleTestError is imported if not already done
from statsmodels.tools.sm_exceptions import InfeasibleTestError

def run_granger_causality(df: pd.DataFrame, var1: str, var2: str, max_lag: int = 12) -> Optional[Dict[int, float]]:
    """
    Performs Granger Causality tests using first differences.
    Attempts test on levels if differenced series is constant (with warning).
    Returns {lag: p_value}.
    """
    logger.info(f"Running Granger Causality test: '{var1}' -> '{var2}'? (Max lag: {max_lag} weeks)")
    if var1 not in df.columns or var2 not in df.columns:
        logger.error(f"Granger columns missing ('{var1}' or '{var2}'). Skipping test.")
        return None

    data = df[[var1, var2]].copy()
    # Drop rows with NaNs *before* checking for constant values
    data.dropna(inplace=True)

    if data.shape[0] < max_lag + 5: # Check after dropping NaNs
        logger.error(f"Not enough data ({data.shape[0]}) for Granger test after dropna. Skipping test.")
        return None

    # Check for constant columns in the original data
    if (data[var1].nunique() <= 1) or (data[var2].nunique() <= 1):
        logger.error(f"Input data for '{var1}' or '{var2}' is constant before differencing. Granger test impossible. Skipping test.")
        return None

    # Proceed with differencing
    try:
        data_diff = data.diff().dropna()
    except Exception as e:
        logger.error(f"Granger differencing error: {e}. Skipping test.")
        return None

    if data_diff.shape[0] < max_lag + 5: # Check after differencing
        logger.error(f"Not enough data ({data_diff.shape[0]}) after differencing. Skipping test.")
        return None

    # Check for constant columns *after* differencing
    test_on_diff = True
    if (data_diff[var1].nunique() <= 1) or (data_diff[var2].nunique() <= 1):
        logger.warning(f"Constant column found after differencing for '{var1}' or '{var2}'.")
        # Attempt test on levels as fallback
        logger.warning(f"Attempting Granger test on raw levels for '{var1}' -> '{var2}' (use results with caution).")
        test_on_diff = False
        test_data = data[[var2, var1]] # Use original data
        if test_data.shape[0] < max_lag + 5: # Re-check length for levels data
             logger.error("Not enough data for levels test. Skipping.")
             return None
    else:
        logger.info("Using first differences for Granger test.")
        test_data = data_diff[[var2, var1]] # Use differenced data


    # Perform the test
    try:
        gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        p_values = {lag: gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}

        # Log results clearly
        significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
        test_type = "(Differences)" if test_on_diff else "(Levels - CAUTION!)"
        if significant_lags:
            logger.info(f" Granger Result ('{var1}' -> '{var2}') {test_type}: Significant at lags: {significant_lags} (p < 0.05).")
        else:
            logger.info(f" Granger Result ('{var1}' -> '{var2}') {test_type}: Not significant up to lag {max_lag} (p >= 0.05).")
        return p_values

    except InfeasibleTestError as e: # Catch specific error
         logger.error(f"Granger test error for '{var1}' -> '{var2}' (Infeasible): {e}. Likely constant values even in levels.")
         return None
    except Exception as e:
        logger.error(f"Granger test error for '{var1}' -> '{var2}': {e}", exc_info=True)
        return None

@memory.cache
def calculate_sentiment_scores_dataframe(df: pd.DataFrame, text_col: str = "text", batch_size: int = 32) -> pd.DataFrame:
    """Calculates multiple MacBERTh sentiment scores for the text column."""
    logger.info(f"Calculating multiple MacBERTh sentiment scores for '{text_col}'...")
    if text_col not in df.columns: raise ValueError(f"Column '{text_col}' not found.")

    df_copy = df.copy()
    df_copy[text_col] = df_copy[text_col].astype(str).fillna('') # Ensure string type

    scorer = MacBERThSentimentScorer() # Uses the class defined above
    texts_to_score = df_copy[text_col].tolist()

    # Get the dictionary of scores {score_name: [list_of_scores]}
    sentiment_scores_dict = scorer.calculate_sentiment_scores(texts_to_score, batch_size=batch_size)

    # Add each score list as a new column
    score_cols_added = []
    for score_name, scores_list in sentiment_scores_dict.items():
        if scores_list: # Only add if scores were calculated
             df_copy[score_name] = scores_list
             score_cols_added.append(score_name) # Keep track of added columns
             logger.info(f"Added sentiment scores for '{score_name}'. Stats: Min={np.min(scores_list):.3f}, Max={np.max(scores_list):.3f}, Mean={np.mean(scores_list):.3f}, Std={np.std(scores_list):.3f}")
        else:
             logger.warning(f"No scores calculated for '{score_name}', column not added.")

    logger.info("Multiple sentiment scoring complete.")
    # Return identifier columns PLUS the newly added score columns
    id_cols = ['week_date', 'doc_id', 'trial_id'] # Keep identifiers
    # Ensure we only select columns that actually exist in the df_copy
    final_cols = [col for col in id_cols + score_cols_added if col in df_copy.columns]
    return df_copy[final_cols]

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

def main():
    """
    Runs analysis pipeline using pre-computed sentiment, aggregates over full period,
    then crops final weekly data for analysis.
    """
    logger.info("--- Starting Historical Analysis Script (Old Bailey - Combined Metrics - Aggregate then Crop) ---")
    script_start_time = time.time()
    final_df_cropped = None # Renamed variable for clarity
    smoothed_structural_col_names_map = { # Mapping used in aggregate func
            'violent_crime_prop': 'violent_crime_prop_smooth', 'property_crime_prop': 'property_crime_prop_smooth',
            'conviction_rate': 'conviction_rate_smooth', 'death_sentence_rate': 'death_sentence_rate_smooth',
            'transport_rate': 'transport_rate_smooth', 'avg_punishment_score': 'avg_punishment_score_smooth'
    }
    best_model = None; trainer = None; val_dl = None; val_ds = None

    try:
        os.makedirs(PLOT_DIR, exist_ok=True); logger.info(f"Plots directory: {PLOT_DIR}")

        # --- DEFINE ANALYSIS PERIOD FOR FINAL CROPPING ---
        ANALYSIS_START_DATE = '1719-01-01' 
        ANALYSIS_END_DATE = '1829-12-31'
        logger.info(f"*** Final analysis will be restricted to date range: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE} ***")
        # ---------------------------------------------

        # --- Step 1: Parse Old Bailey Data (Structured + Text) ---
        # Still need the full parsed data for aggregation input
        logger.info("--- Step 1: Parsing Old Bailey Sessions Papers (Full Period) ---")
        ob_df_parsed = parse_old_bailey_papers(ob_dir=OLD_BAILEY_DIR, start_year=START_YEAR, end_year=END_YEAR)
        if ob_df_parsed.empty: raise ValueError("Step 1 Failed.")
        if 'text' not in ob_df_parsed.columns: raise ValueError("Parsing missing 'text'.")

        # --- Step 2: Calculate Sentiment Scores (Uses Cache if Run Before) ---
        # Run this on the full parsed data so it's available for aggregation
        logger.info("--- Step 2: Calculating/Loading Targeted Sentiment Scores per Trial (Full Period) ---")
        ob_df_sentiment_scores = calculate_sentiment_scores_dataframe(ob_df_parsed, text_col='text', batch_size=BATCH_SIZE // 2)
        if ob_df_sentiment_scores.empty: logger.warning("Step 2 Warning: Sentiment scoring resulted in empty DataFrame.")
        else: logger.info(f"Sentiment scores calculated/loaded. Columns: {ob_df_sentiment_scores.columns.tolist()}")

        # --- Step 3: Load Weekly Mortality (Full Period) ---
        logger.info("--- Step 3: Loading and Processing Weekly Mortality Data (Full Period) ---")
        weekly_mortality_df = load_and_aggregate_weekly_mortality(file_path=COUNTS_FILE, start_year=START_YEAR, end_year=END_YEAR)
        if weekly_mortality_df.empty: raise ValueError("Step 3 Failed.")

        # --- Step 4: Aggregate Combined Metrics (OVER FULL PERIOD) ---
        logger.info(f"--- Step 4: Aggregating Weekly Combined Metrics and Merging (Full Period) ---")
        # The aggregate function itself performs clipping at the end of ITS process
        final_df_full = aggregate_weekly_combined_metrics(ob_df_parsed, ob_df_sentiment_scores, weekly_mortality_df)
        if final_df_full is None or final_df_full.empty: raise ValueError("Step 4 Failed.")
        logger.info(f"Full aggregated DataFrame shape (before final crop): {final_df_full.shape}")
        del ob_df_parsed; del ob_df_sentiment_scores; del weekly_mortality_df

        # --- Step 5: CROP the Aggregated DataFrame to Analysis Period ---
        logger.info(f"--- Step 5: Cropping Aggregated Data to {ANALYSIS_START_DATE} - {ANALYSIS_END_DATE} ---")
        if 'week_date' not in final_df_full.columns:
             raise ValueError("Aggregation failed to produce 'week_date' column.")
        final_df_cropped = final_df_full[
            (final_df_full['week_date'] >= pd.to_datetime(ANALYSIS_START_DATE)) &
            (final_df_full['week_date'] <= pd.to_datetime(ANALYSIS_END_DATE))
        ].copy() # Create a copy to avoid SettingWithCopyWarning later

        # Check if data remains after cropping
        if final_df_cropped.empty:
             raise ValueError(f"No data remains after cropping final aggregated DataFrame to {ANALYSIS_START_DATE}-{ANALYSIS_END_DATE}. Check dates or aggregation results.")

        # Optional: Reset time_idx for the cropped period if desired, needed if TFT requires 0-based index
        # final_df_cropped['time_idx'] = (final_df_cropped['week_date'] - final_df_cropped['week_date'].min()).dt.days // 7
        # logger.info("Re-indexed time_idx for cropped period.")

        logger.info(f"Cropped final DataFrame shape: {final_df_cropped.shape}")
        logger.info(f"Cropped final DataFrame time index range: {final_df_cropped['time_idx'].min()} - {final_df_cropped['time_idx'].max()}")
        del final_df_full # Free memory

        # --- Step 5.5: Save Cropped Data ---
        try:
            save_filename = f"final_weekly_data_combined_metrics_logdeaths_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}_agg_then_crop.csv"
            final_df_cropped.to_csv(os.path.join(PLOT_DIR, save_filename), index=False)
            logger.info(f"Saved CROPPED final weekly data for period {ANALYSIS_START_DATE[:4]}-{ANALYSIS_END_DATE[:4]}.")
        except Exception as e: logger.warning(f"Could not save final data CSV: {e}")

        # --- Step 6: Generate Plots (using CROPPED final_df) ---
        logger.info("--- Step 6: Generating Data Exploration Plots (Cropped Period) ---")
        # Define columns based on expected output
        viol_prop_smooth_col = 'violent_crime_prop_smooth'; conv_rate_smooth_col = 'conviction_rate_smooth';
        disease_sent_col = 'disease_sentiment'; hardship_sent_col = 'hardship_sentiment'
        lag_hardship_sent_col = f'hardship_sentiment_lag{LAG_WEEKS}w'
        hardship_sent_std_col = 'hardship_sentiment_std'
        all_possible_sent_cols = list(MacBERThSentimentScorer._word_lists.keys())
        total_trials_col = 'total_trials'

        # All plotting now uses final_df_cropped
        if final_df_cropped is not None and not final_df_cropped.empty:
            plot_time_series(final_df_cropped, 'week_date', 'deaths', 'Weekly Deaths Over Time (Original)', 'Total Deaths', 'deaths_weekly_timeseries_original.png', PLOT_DIR)
            if 'log_deaths' in final_df_cropped.columns: plot_time_series(final_df_cropped, 'week_date', 'log_deaths', 'Weekly Log(Deaths+1) Over Time', 'Log(Deaths+1)', 'deaths_weekly_timeseries_log.png', PLOT_DIR)
            if viol_prop_smooth_col in final_df_cropped.columns: plot_time_series(final_df_cropped, 'week_date', viol_prop_smooth_col, 'Weekly Violent Crime Prop (Smoothed)', 'Proportion of Trials', f'{viol_prop_smooth_col}_timeseries.png', PLOT_DIR)
            if conv_rate_smooth_col in final_df_cropped.columns: plot_time_series(final_df_cropped, 'week_date', conv_rate_smooth_col, 'Weekly Conviction Rate (Smoothed)', 'Rate', f'{conv_rate_smooth_col}_timeseries.png', PLOT_DIR)

            sentiment_zoom = (0.60, 0.85)
            for sent_col in all_possible_sent_cols:
                if sent_col in final_df_cropped.columns:
                    plot_title = f"Weekly Avg. {sent_col.replace('_', ' ').title()}"
                    plot_filename = f"{sent_col}_timeseries_zoomed.png"
                    plot_time_series(final_df_cropped, 'week_date', sent_col, plot_title, 'Avg. Score', plot_filename, PLOT_DIR, zoom_ylim=sentiment_zoom, mark_zero_threshold=0.1)
                else: logger.info(f"Sentiment column '{sent_col}' not found, skipping plot.")

            if total_trials_col in final_df_cropped.columns:
                plot_time_series(final_df_cropped, 'week_date', total_trials_col,'Weekly Total Old Bailey Trials Processed', 'Number of Trials', 'total_trials_weekly_timeseries.png', PLOT_DIR)

            if 'log_deaths' in final_df_cropped.columns and hardship_sent_std_col in final_df_cropped.columns:
                plot_dual_axis(final_df_cropped, 'week_date', 'log_deaths', hardship_sent_std_col, 'Log(Deaths+1)', 'Std. Hardship Sent.', 'Log(Deaths+1) vs Std. Hardship Sentiment', 'logdeaths_vs_hardship_sent_std_dual_axis.png', PLOT_DIR)
            if 'log_deaths' in final_df_cropped.columns and lag_hardship_sent_col in final_df_cropped.columns:
                plot_scatter_fear_vs_deaths(final_df_cropped, lag_hardship_sent_col, 'log_deaths', f'Log(Deaths+1) vs Lagged ({LAG_WEEKS}w) Hardship Sentiment', f'scatter_logdeaths_vs_{lag_hardship_sent_col}.png', PLOT_DIR)
            plot_weekly_boxplot(final_df_cropped, 'deaths', 'Weekly Distribution of Deaths (by Week of Year - Original Scale)', 'boxplot_deaths_weekly_original.png', PLOT_DIR)
            # --- ADD NEW PLOTS ---
            logger.info("--- Generating Additional Diagnostic Plots ---")
            plot_lags = 20 # How many lags for CCF/ACF/PACF
            rolling_window_years = 5
            rolling_window_weeks = rolling_window_years * 52

            target_col = 'log_deaths'
            lag_to_plot = 4 # lag 4 based on CCF/Granger significance

            # Lag Scatter: Hardship Sentiment(t-k) vs LogDeaths(t)
            if target_col in final_df_cropped.columns and hardship_sent_col in final_df_cropped.columns:
                plot_lag_scatter(final_df_cropped, target_col, hardship_sent_col, lag=lag_to_plot,
                                 title=f"Log(Deaths+1) vs Lagged Hardship Sentiment",
                                 filename=f"lag_scatter_{target_col}_vs_{hardship_sent_col}_lag{lag_to_plot}w.png",
                                 plot_dir=PLOT_DIR)
            else:
                 logger.warning(f"Skipping Lag Scatter: Missing '{hardship_sent_col}' or '{target_col}'.")

            # Lag Scatter: Property Crime Prop Smooth(t-k) vs LogDeaths(t)
            if prop_crime_smooth and target_col in final_df_cropped.columns and prop_crime_smooth in final_df_cropped.columns:
                plot_lag_scatter(final_df_cropped, target_col, prop_crime_smooth, lag=lag_to_plot,
                                 title=f"Log(Deaths+1) vs Lagged Property Crime Prop (Smooth)",
                                 filename=f"lag_scatter_{target_col}_vs_{prop_crime_smooth}_lag{lag_to_plot}w.png",
                                 plot_dir=PLOT_DIR)
            else:
                 logger.warning(f"Skipping Lag Scatter: Missing '{prop_crime_smooth}' or '{target_col}'.")

            # CCF Plots (Example: LogDeaths vs Hardship Sentiment)
            if 'log_deaths' in final_df_cropped.columns and hardship_sent_col in final_df_cropped.columns:
                plot_ccf(final_df_cropped, 'hardship_sentiment', 'log_deaths', plot_lags,
                         f"CCF: Hardship Sentiment vs Log(Deaths)",
                         "ccf_hardship_vs_logdeaths.png", PLOT_DIR)
                
                            # CCF Plots (Example: LogDeaths vs Hardship Sentiment)
            if 'log_deaths' in final_df_cropped.columns and disease_sent_col in final_df_cropped.columns:
                plot_ccf(final_df_cropped, 'disease_sentiment', 'log_deaths', plot_lags,
                         f"CCF: Hardship Sentiment vs Log(Deaths)",
                         "ccf_hardship_vs_logdeaths.png", PLOT_DIR)

            # Rolling Correlation (Example: LogDeaths vs Property Crime Prop Smooth)
            prop_crime_smooth = smoothed_structural_col_names_map.get('property_crime_prop')
            if prop_crime_smooth and 'log_deaths' in final_df_cropped.columns and prop_crime_smooth in final_df_cropped.columns:
                 plot_rolling_correlation(final_df_cropped, 'week_date', 'log_deaths', prop_crime_smooth,
                                         window=rolling_window_weeks,
                                         title=f"{rolling_window_years}y Rolling Correlation: LogDeaths vs Property Crime Prop (Smooth)",
                                         filename=f"roll_corr_logdeaths_vs_{prop_crime_smooth}.png", plot_dir=PLOT_DIR)

            # ACF/PACF Plots
            if 'log_deaths' in final_df_cropped.columns:
                plot_acf_pacf(final_df_cropped['log_deaths'], plot_lags, "Log(Deaths+1)", "log_deaths", PLOT_DIR)
            if hardship_sent_col in final_df_cropped.columns:
                 plot_acf_pacf(final_df_cropped[hardship_sent_col], plot_lags, "Hardship Sentiment", "hardship_sentiment", PLOT_DIR)

            # --- END ADD NEW PLOTS ---

        else: logger.warning("Cropped final_df empty, skipping data plots.")


        # === Step 7: Train TFT Model (using CROPPED final_df) ===
        run_name = f"combined_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}"
        logger.info(f"--- Step 7: Training and Evaluating TFT Model ({run_name}) ---")
        if final_df_cropped is not None and not final_df_cropped.empty:
            # Get standardized features from the cropped dataframe
            tft_real_features_exist = [col for col in final_df_cropped.columns if col.endswith('_std')]
            logger.info(f"Using these standardized features for TFT ({run_name}): {tft_real_features_exist}")

            if not tft_real_features_exist: logger.error("No standardized features found in cropped df. Skipping training.")
            else:
                # Train on the cropped data
                best_model, trainer, val_dl, val_ds = train_tft_model(
                    df=final_df_cropped, # Pass the cropped dataframe
                    time_varying_reals_cols=tft_real_features_exist,
                    run_name=run_name,
                    max_epochs=75,
                )

                # Evaluate Model
                if best_model and val_dl and val_ds:
                    logger.info(f"--- Evaluating Best Model for Run: {run_name} ---")
                    eval_metrics = evaluate_model(best_model, val_dl, val_ds, PLOT_DIR, run_name=run_name)
                    logger.info(f"Final Validation Metrics ({run_name}): {eval_metrics}")

                    # Plot Training History
                    if trainer and hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
                         logger.info(f"--- Plotting Training History for Run: {run_name} ---")
                         plot_training_history(trainer.logger.log_dir, run_name, PLOT_DIR)
                else: logger.error(f"TFT Model training/loading failed for run '{run_name}'.")
        else: logger.error("Cropped Final DataFrame empty, cannot train TFT model.")


        # === Step 8: Interpretation & Granger Causality (using CROPPED final_df) ===
        logger.info("--- Step 8: Model Interpretation & Granger Causality (Cropped Period) ---")
        if final_df_cropped is not None and not final_df_cropped.empty:
             # Interpretation
            if best_model and val_dl:
                logger.info(f"--- Running TFT Interpretation ({run_name}) ---")
                interpret_tft(best_model, val_dl, PLOT_DIR, run_name=run_name)
            else: logger.warning("Skipping TFT interpretation.")

            # Granger Causality
            logger.info("--- Running Granger Causality Tests (Cropped Period) ---")
            # Granger runs on the final cropped dataframe
            granger_results = {}
            max_weekly_lag = 12
            target_col_granger = 'log_deaths'
            # Define base names again for clarity (even though derived from class)
            sentiment_cols_agg_base = list(MacBERThSentimentScorer._word_lists.keys())
            lagged_cols_base = [smoothed_structural_col_names_map.get('conviction_rate'), 'hardship_sentiment'] # Use smooth name
            lagged_cols_base = [col for col in lagged_cols_base if col is not None]

            metrics_to_test_granger = {}
            for base_col, smooth_col in smoothed_structural_col_names_map.items(): metrics_to_test_granger[smooth_col] = f"{base_col.replace('_', ' ').title()} (Smooth)"
            for base_col in sentiment_cols_agg_base: metrics_to_test_granger[base_col] = base_col.replace('_', ' ').title()
            for base_col in lagged_cols_base:
                 lag_col = f"{base_col}_lag{LAG_WEEKS}w"
                 if lag_col in final_df_cropped.columns:
                     label_base = base_col.replace('_smooth','').replace('_', ' ').title()
                     metrics_to_test_granger[lag_col] = f"Lag {LAG_WEEKS}w {label_base}"

            metrics_to_test_granger_final = {k: v for k, v in metrics_to_test_granger.items() if k in final_df_cropped.columns}
            logger.info(f"Metrics selected for Granger testing (Cropped): {list(metrics_to_test_granger_final.keys())}")

            for metric_col, metric_label in metrics_to_test_granger_final.items():
                if target_col_granger in final_df_cropped.columns:
                    logger.info(f"Granger tests: '{metric_col}' vs '{target_col_granger}'")
                    desc1 = f"'{metric_label}' -> 'Log(Deaths)'"
                    granger_results[desc1] = run_granger_causality(final_df_cropped, metric_col, target_col_granger, max_lag=max_weekly_lag)
                    desc2 = f"'Log(Deaths)' -> '{metric_label}'"
                    granger_results[desc2] = run_granger_causality(final_df_cropped, target_col_granger, metric_col, max_lag=max_weekly_lag)
                else: logger.warning(f"Target '{target_col_granger}' missing."); break

            if granger_results: plot_granger_causality_results(granger_results, title_prefix=f"Cropped_{ANALYSIS_START_DATE[:4]}_{ANALYSIS_END_DATE[:4]}_Metrics_vs_LogDeaths", plot_dir=PLOT_DIR)
            else: logger.warning("No valid Granger results to plot for cropped data.")

        else: logger.warning("Cropped Final DataFrame empty, skipping interpretation & Granger.")

    # --- Error Handling & Script End ---
    except FileNotFoundError as e: logger.error(f"Data file not found: {e}.")
    except ValueError as e: logger.error(f"Data processing/config error: {e}", exc_info=True)
    except ImportError as e: logger.error(f"Missing library: {e}.")
    except Exception as e: logger.error(f"Unexpected script error: {e}", exc_info=True)

    script_end_time = time.time()
    logger.info(f"--- Script finished in {(script_end_time - script_start_time):.2f} seconds ({(script_end_time - script_start_time)/60:.2f} minutes) ---")

# --- Run Main ---
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()