import pandas as pd
import numpy as np
import ast
import os
import warnings
import logging
import sys
import io
import re
import traceback
from typing import List, Tuple, Set, Optional, Union

from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Environment Setup ---
def setup_environment():
    """Configures the environment for silent and correct operation on Windows and with HF models."""
    if sys.platform == "win32":
        # Ensure UTF-8 output for console icons/colors
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Aggressive Noise Suppression for Transformers and HF
    os.environ.update({
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING": "1"
    })

    # Silence standard logging
    logging.basicConfig(level=logging.ERROR)
    for logger_name in ["transformers", "sentence_transformers", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    warnings.filterwarnings("ignore")

    try:
        import transformers
        import huggingface_hub
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.disable_propagation()
        huggingface_hub.logging.set_verbosity_error()
    except ImportError:
        pass

setup_environment()

class Colors:
    """ANSI color codes for pretty console output."""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    PURPLE = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"

class BoardGameRecommender:
    """
    A recommendation system for board games using SBERT and numerical features.
    """
    
    NUMERICAL_FEATURES = ['min_players', 'max_players', 'min_playtime', 'max_playtime', 'min_age', 'weight']

    def __init__(self, 
                 data_path: str = 'content-based/data/bgg_games_cleaned.csv', 
                 raw_data_path: str = 'content-based/data/bgg_boardgames.csv',
                 bi_encoder_name: str = 'BAAI/bge-base-en-v1.5'):
        """
        Initializes the recommender by loading data and generating embeddings.
        """
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.bi_encoder_name = bi_encoder_name
        
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.scaled_numerical: Optional[np.ndarray] = None
        self.id_to_idx: Dict[int, int] = {}
        
        # BGE requires a specific instruction for retrieval (query-side only)
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        # 1. Load Bi-Encoder model
        print(f"{Colors.BLUE}[*] Loading Bi-Encoder: {self.bi_encoder_name}...{Colors.END}")
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        
        # 2. Data Preparation
        self._load_data()
        self._preprocess_data()
        self._generate_embeddings()

    @staticmethod
    def _parse_to_string(x) -> str:
        """Helper to convert list-like strings from CSV into clean comma-separated strings."""
        if pd.isna(x):
            return ""
        try:
            if isinstance(x, str) and (x.startswith('[') or x.startswith('{')):
                items = ast.literal_eval(x)
                if isinstance(items, list):
                    return ', '.join(map(str, items))
                return str(items)
            return str(x)
        except (ValueError, SyntaxError):
            return str(x)

    def _load_data(self):
        """Loads processed and raw dataframes."""
        print(f"[*] Loading processed data from {Colors.BOLD}{os.path.basename(self.data_path)}{Colors.END}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Could not find data file at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.id_to_idx = {int(bid): idx for idx, bid in enumerate(self.df['id'])}
        
        if os.path.exists(self.raw_data_path):
            print(f"[*] Loading raw data for display from {Colors.BOLD}{os.path.basename(self.raw_data_path)}{Colors.END}...")
            self.raw_df = pd.read_csv(self.raw_data_path)
        else:
            print(f"{Colors.YELLOW}[!] Raw data file not found. Using processed data for display.{Colors.END}")
            self.raw_df = self.df.copy()

    def _preprocess_data(self):
        """Prepares semantic search text and scales numerical features for distance calculation."""
        print("[*] Preprocessing content for recommendations...")
        
        # Construct the semantic context string for the Bi-Encoder
        self.df['semantic_text'] = self.df.apply(
            lambda x: (
                f"Game: {x['name']}. "
                f"Designers: {self._parse_to_string(x['designers'])}. "
                f"Categories: {self._parse_to_string(x['categories'])}. "
                f"Mechanics: {self._parse_to_string(x['mechanics'])}. "
                f"Families: {self._parse_to_string(x['families'])}. "
                f"Description: {str(x['description'])}"
            ), axis=1
        ).str.lower()

        # Ensure numerical features are clean
        for col in self.NUMERICAL_FEATURES:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col].fillna(self.df[col].median(), inplace=True)
                    
        scaler = StandardScaler()
        self.scaled_numerical = scaler.fit_transform(self.df[self.NUMERICAL_FEATURES])

    def _generate_embeddings(self):
        """Generates SBERT embeddings or loads them from a local cache file."""
        cache_name = self.bi_encoder_name.replace('/', '_')
        embedding_cache = self.data_path.replace('.csv', f'_{cache_name}_embeddings.npy')
        
        if os.path.exists(embedding_cache):
            print(f"[*] Loading cached embeddings from {Colors.BOLD}{os.path.basename(embedding_cache)}{Colors.END}...")
            try:
                self.embeddings = np.load(embedding_cache)
                # Quick validation to ensure it matches dataset size (assumes 2D array)
                if self.embeddings.shape[0] != len(self.df):
                    raise ValueError("Cached embeddings shape does not match the dataset size.")
            except Exception as e:
                print(f"{Colors.YELLOW}[!] Warning: Cached embeddings are corrupted or invalid ({e}). Regenerating...{Colors.END}")
                self.embeddings = None

        if self.embeddings is None:
            print(f"{Colors.YELLOW}[!] Generating new embeddings (this may take a few minutes)...{Colors.END}")
            self.embeddings = self.bi_encoder.encode(
                self.df['semantic_text'].tolist(), 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            try:
                # Save as float16 to save disk space if possible, or just regular float32
                np.save(embedding_cache, self.embeddings.astype(np.float32))
            except OSError as e:
                print(f"{Colors.YELLOW}[!] Warning: Could not save embeddings to disk ({e}). Your drive might be out of space. Proceeding without caching.{Colors.END}")
                # Clean up the potentially corrupted partial file
                if os.path.exists(embedding_cache):
                    try:
                        os.remove(embedding_cache)
                    except Exception:
                        pass

    def _get_family_tags(self, families_str: str, prefix: str) -> Set[str]:
        """Parses the 'families' field to extract specific tags (e.g., 'Game: ', 'Series: ')."""
        try:
            tags = ast.literal_eval(families_str)
            return {t.split(': ')[1].lower() for t in tags if t.startswith(prefix)}
        except (ValueError, SyntaxError):
            return set()

    def _filter_family_games(self, target_indices: List[int], candidates_df: pd.DataFrame) -> Tuple[List[int], List[str]]:
        """
        Intelligent filter to exclude expansions and literal variants (same 'Game' tag) 
        while preserving sibling games in the same series (same 'Series' tag but different 'Game' tag).
        """
        target_names = []
        target_games = set()
        target_series = set()
        
        # Aggregate tags for all target games
        for target_idx in target_indices:
            target_row = self.df.loc[target_idx]
            target_names.append(target_row['name'].lower())
            target_games.update(self._get_family_tags(target_row['families'], "Game: "))
            target_series.update(self._get_family_tags(target_row['families'], "Series: "))
        
        filtered_indices = []
        excluded_names = []

        for row in candidates_df.itertuples():
            cand_name = row.name.lower()
            cand_families = str(row.families)
            
            cand_games = self._get_family_tags(cand_families, "Game: ")
            cand_series = self._get_family_tags(cand_families, "Series: ")
            
            # Logic conditions:
            # 1. Is it a variant/expansion? (Shares the exact 'Game' family tag with any target)
            is_variant = bool(target_games and cand_games and (target_games & cand_games))
            
            # 2. Is it a sibling in a series? (Shares a 'Series' tag, but not the same 'Game' tag)
            is_sibling = bool(target_series and cand_series and (target_series & cand_series))
            
            # 3. Simple name overlap fallback for any target
            is_prefix_overlap = any(cand_name.startswith(t_name) for t_name in target_names)

            # RULE: Exclude if it's a direct variant OR has a direct name prefix,
            # UNLESS it is identified as a distinct 'Series' sibling.
            should_exclude = (is_variant or is_prefix_overlap) and not (is_sibling and not is_variant)

            if not should_exclude:
                filtered_indices.append(row.Index)
            else:
                excluded_names.append(row.name)
                
        return filtered_indices, sorted(list(set(excluded_names)))

    def get_recommendations(self, game_names: Union[str, List[str]], n: int = 5, alpha: float = 0.5, beta: float = 0.5, diverse: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Computes hybrid recommendations combining semantic similarity and numerical feature similarity.
        
        Args:
            game_names: The name of the board game (or list of names) to find similarities for.
            n: Number of recommendations to return.
            alpha: Weight for the semantic (SBERT) similarity score.
            beta: Weight for the numerical feature similarity score.
            diverse: If true, filters out expansions and close variants.
            
        Returns:
            A tuple of (Results DataFrame, List of excluded game names).
        """
        # Normalize to list
        if isinstance(game_names, str):
            game_names = [game_names]
            
        target_indices = []
        for g_name in game_names:
            mask = self.df['name'].str.lower() == g_name.lower()
            if not mask.any():
                print(f"{Colors.YELLOW}[!] Game '{g_name}' not found. Skipping it.{Colors.END}")
                continue
            target_indices.append(self.df[mask].index[0])
            
        if not target_indices:
            print(f"{Colors.RED}[!] No valid games found in input. Please check spelling.{Colors.END}")
            return pd.DataFrame(), []
            
        # Calculate combined vectors (average)
        # For BGE, we need to re-encode the combined query_text with an instruction
        combined_text = ". ".join(self.df.loc[target_indices, 'semantic_text'].tolist())
        query_text = self.query_instruction + combined_text
        query_emb = self.bi_encoder.encode(query_text, convert_to_numpy=True)
        
        # 1. Semantic Similarity (Cosine)
        semantic_sim = np.clip(cosine_similarity([query_emb], self.embeddings).flatten(), 0, 1)
        
        # 2. Numerical Similarity (Euclidean Distance based)
        target_num = np.mean(self.scaled_numerical[target_indices], axis=0).reshape(1, -1)
        dist = np.linalg.norm(self.scaled_numerical - target_num, axis=1)
        # Normalize distance to [0,1] similarity; cap distance at 10 for normalization
        num_sim = 1 - (np.clip(dist, 0, 10) / 10)
        
        # 3. Hybrid Score
        hybrid_scores = (alpha * semantic_sim) + (beta * num_sim)
        
        # 4. Broad Retrieval (top 1000 candidates for filtering)
        # Exclude ALL target games themselves from the results
        top_indices = [i for i in hybrid_scores.argsort()[::-1] if i not in target_indices][:1000]
        candidates = self.df.iloc[top_indices].copy()
        candidates['sem_score'] = semantic_sim[top_indices]
        candidates['num_score'] = num_sim[top_indices]
        candidates['hyb_score'] = hybrid_scores[top_indices]
        
        # 5. Diversity / Family Filtering
        excluded_names = []
        if diverse:
            filtered_idx, excluded_names = self._filter_family_games(target_indices, candidates)
            candidates = candidates.loc[filtered_idx].copy()

        # 6. Selection
        result = candidates.sort_values('hyb_score', ascending=False).head(n).copy()
            
        # 7. Metadata Enrichment (Merge back with Raw Data for display purposes)
        if 'id' in self.df.columns and 'id' in self.raw_df.columns:
            display_cols = ['id', 'name', 'year', 'image', 'description']
            available_cols = [c for c in display_cols if c in self.raw_df.columns]
            
            enrichment_df = self.raw_df[self.raw_df['id'].isin(result['id'])][available_cols]
            result = result.merge(enrichment_df, on='id', suffixes=('_proc', ''), how='left')
            
            # Prefer raw metadata where available
            for col in ['name', 'year', 'description']:
                if f'{col}_proc' in result.columns:
                    result[col] = result[col].fillna(result[f'{col}_proc'])
            
            result = result.sort_values('hyb_score', ascending=False)
        
        # Final output columns
        final_cols = ['name', 'year', 'sem_score', 'num_score', 'hyb_score']
        return result[final_cols].reset_index(drop=True), excluded_names

def print_results(target: Union[str, List[str]], df: pd.DataFrame, excluded_names: List[str] = None):
    """Prints recommendations in a high-fidelity console table."""
    if df.empty:
        return

    if isinstance(target, list):
        display_target = ", ".join(target)
    else:
        display_target = target

    width = 90
    print(f"\n{Colors.PURPLE}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}TOP RECOMMENDATIONS FOR:{Colors.END} {Colors.GREEN}{display_target[:55]}{'...' if len(display_target) > 55 else ''}{Colors.END}")
    print(f"{Colors.PURPLE}{'='*width}{Colors.END}")
    
    header = f"{'#':<3} | {'Game Name':<45} | {'Year':<6} | {'Sem':<5} | {'Num':<5} | {'Hyb':<5}"
    print(header)
    print("-" * width)
    
    for i, row in enumerate(df.itertuples(), 1):
        year = str(int(row.year)) if pd.notna(row.year) else "N/A"
        name = str(row.name)[:45]
        print(f"{i:<3} | {name:<45} | {year:<6} | {row.sem_score:<5.2f} | {row.num_score:<5.2f} | {row.hyb_score:<5.2f}")
    
    if excluded_names:
        print(f"{Colors.PURPLE}{'-'*width}{Colors.END}")
        title = f"EXCLUDED {len(excluded_names)} FAMILY VARIANT(S) / EXPANSIONS"
        print(f"{Colors.YELLOW}{title:^{width}}{Colors.END}")
        print(f"{Colors.PURPLE}{'-'*width}{Colors.END}")
        
        # Print excluded games in a multi-column grid
        cols = 3
        col_width = (width - 6) // cols
        max_display = 30
        for i in range(0, min(len(excluded_names), max_display), cols):
            row_items = excluded_names[i:i+cols]
            row_str = " | ".join(f"{str(item)[:col_width]:<{col_width}}" for item in row_items)
            print(f" {row_str}")
        
        if len(excluded_names) > max_display:
            print(f" ... and {len(excluded_names) - max_display} more.")
            
    print(f"{Colors.PURPLE}{'='*width}{Colors.END}\n")

def main():
    """Main execution function with easy configuration."""
    # ==========================================
    # ⚙️ USER SETTINGS
    # ==========================================
    TEST_QUERY = ["Point City", "Point Salad"]  # Games to find similarities for
    TOP_N = 10                                  # Number of recommendations to show
    # ==========================================

    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'bgg_games_cleaned.csv')
    
    if not os.path.exists(data_file):
        print(f"{Colors.RED}[!] Data file not found at {data_file}{Colors.END}")
        return

    try:
        recommender = BoardGameRecommender(data_path=data_file)
        
        results, excluded = recommender.get_recommendations(TEST_QUERY, n=TOP_N, diverse=True)
        print_results(TEST_QUERY, results, excluded)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[!] Process interrupted by user.{Colors.END}")
    except Exception:
        print(f"{Colors.RED}[!] A critical error occurred:{Colors.END}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
