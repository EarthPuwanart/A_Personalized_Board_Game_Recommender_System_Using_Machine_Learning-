import os
import pandas as pd
import numpy as np
import sys
import traceback
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Import existing recommenders
# Note: Since they are in different subdirectories, we need to handle paths
sys.path.append(os.path.join(os.getcwd(), 'collaborative'))
sys.path.append(os.path.join(os.getcwd(), 'content-based'))

try:
    from svd_recommender import BoardGameSVD, Colors
    from recommender import BoardGameRecommender
except ImportError:
    # Manual path fix if run from different dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, 'collaborative'))
    sys.path.append(os.path.join(script_dir, 'content-based'))
    from svd_recommender import BoardGameSVD, Colors
    from recommender import BoardGameRecommender

class HybridRecommender:
    def __init__(self, svd_model_path=None, content_data_path=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = self.script_dir
        
        # 1. Initialize SVD Model
        if svd_model_path is None:
            svd_model_path = os.path.join(script_dir, "collaborative", "svd_model.joblib")
        
        if not os.path.exists(svd_model_path):
             print(f"{Colors.YELLOW}[!] SVD model not found. Please train it first using svd_recommender.py{Colors.END}")
             self.svd = None
        else:
             self.svd = BoardGameSVD.load(svd_model_path)

        # 2. Initialize Content-based Model
        if content_data_path is None:
            content_data_path = os.path.join(script_dir, "content-based", "data", "bgg_games_cleaned.csv")
            content_raw_path = os.path.join(script_dir, "content-based", "data", "bgg_boardgames.csv")
        
        print(f"{Colors.BLUE}[*] Initializing Content-based Recommender...{Colors.END}")
        self.content = BoardGameRecommender(data_path=content_data_path, raw_data_path=content_raw_path)

    def reload_svd(self, svd_model_path=None):
        if svd_model_path is None:
            svd_model_path = os.path.join(self.script_dir, "collaborative", "svd_model.joblib")
        
        print(f"{Colors.GREEN}[*] Reloading SVD Model for real-time updates...{Colors.END}")
        if os.path.exists(svd_model_path):
            self.svd = BoardGameSVD.load(svd_model_path)
            return True
        return False

    def recommend(self, username: str, n: int = 10, svd_weight: float = 0.5, content_weight: float = 0.5, filters: Dict = None):
        if not self.svd:
            print(f"{Colors.RED}[!] Hybrid recommendation requires a trained SVD model.{Colors.END}")
            return None

        print(f"\n{Colors.PURPLE}🧠 GENERATING HYBRID RECOMMENDATIONS FOR: {Colors.BOLD}{username}{Colors.END}")
        
        # --- Phase 0: Global Filtering (Full Pool) ---
        # We filter the entire dataset FIRST to ensure we never miss a game that matches criteria
        if filters:
            filtered_pool_df = self._apply_filters(self.content.df, filters)
            print(f"   {Colors.BLUE}[*]{Colors.END} Filtered pool: {len(filtered_pool_df)} / {len(self.content.df)} games match criteria.")
            # If no games match, return empty
            if filtered_pool_df.empty:
                return pd.DataFrame()
            valid_bgg_ids = set(filtered_pool_df['id'].tolist())
        else:
            valid_bgg_ids = None

        # --- Phase 1: Collaborative Filtering (SVD) ---
        print(f"   {Colors.BLUE}[*]{Colors.END} Calculating SVD/Cold-start Scores...")
        
        is_known_to_svd = self.svd and username in self.svd.user_enc.classes_
        
        # Initialize svd_scores aligned with self.content.df
        full_svd_scores = np.full(len(self.content.df), -5.0) # Default very low
        
        if self.svd:
            if is_known_to_svd:
                u_idx = self.svd.user_enc.transform([username])[0]
                model_scores = self.svd.global_mean + self.svd.user_biases[u_idx] + self.svd.item_biases + (self.svd.game_factors @ self.svd.user_factors[u_idx])
                
                # Filter out rated indices from model_scores
                rated_indices = self.svd._sparse.getrow(u_idx).indices
                model_scores[rated_indices] = -1.0
            else:
                model_scores = self.svd.popularity.copy()
            
            # Map model_scores (which match self.svd_id_list) to full_svd_scores (which match self.content.df)
            # Find indices in self.content.df for each game in self.svd_id_list
            # We already have self.svd_id_set for fast checking
            for i, bgg_id in enumerate(self.svd.game_enc.classes_):
                bgg_id_int = int(bgg_id)
                if bgg_id_int in self.content.id_to_idx:
                    full_svd_scores[self.content.id_to_idx[bgg_id_int]] = model_scores[i]
        
        svd_scores = full_svd_scores

        # --- Phase 2: User History & Global Rated Filtering ---
        all_rated_bgg_ids = set()
        if is_known_to_svd:
            rated_indices = self.svd._sparse.getrow(u_idx).indices
            all_rated_bgg_ids.update([int(self.svd.game_enc.inverse_transform([r])[0]) for r in rated_indices])
        
        liked_bgg_ids = []
        real_ratings_path = os.path.join(self.script_dir, "real_user_ratings.csv")
        if os.path.exists(real_ratings_path):
            try:
                real_df = pd.read_csv(real_ratings_path)
                user_all_ratings = real_df[real_df['Username'] == username]
                all_rated_bgg_ids.update(user_all_ratings['BGGId'].astype(int).tolist())
                user_liked = user_all_ratings[user_all_ratings['Rating'] >= 7.0]
                liked_bgg_ids.extend(user_liked['BGGId'].astype(int).tolist())
            except Exception as e:
                print(f"   {Colors.YELLOW}[!] Error reading real_user_ratings.csv: {e}{Colors.END}")

        liked_bgg_ids = list(set(liked_bgg_ids))
        liked_indices = self.content.df[self.content.df['id'].isin(liked_bgg_ids)].index
        
        user_seed_emb = None
        if not liked_indices.empty:
            seed_embeddings = self.content.embeddings[liked_indices]
            user_seed_emb = np.mean(seed_embeddings, axis=0)

        # --- Phase 3: Candidate Selection (Pre-filtered) ---
        # 1. Top SVD candidates that pass filters
        if valid_bgg_ids is not None:
            # Mask SVD scores for invalid games
            filter_mask = self.content.df['id'].isin(valid_bgg_ids).values
            svd_scores[~filter_mask] = -10.0 # Very low score for filtered out games
        
        # Get top 300 candidates
        top_svd_idx = np.argpartition(svd_scores, -300)[-300:]
        
        # Get BGGIds from top_svd_idx (which are indices into self.content.df)
        candidate_bgg_ids = [int(self.content.df.iloc[idx]['id']) for idx in top_svd_idx if svd_scores[idx] > -5]

        # 2. Top Content candidates that pass filters
        content_bgg_ids = []
        if user_seed_emb is not None:
            content_sims = np.clip(cosine_similarity([user_seed_emb], self.content.embeddings).flatten(), 0, 1)
            if valid_bgg_ids is not None:
                content_sims[~filter_mask] = -1.0
            
            top_content_idx = np.argpartition(content_sims, -300)[-300:]
            content_bgg_ids = self.content.df.iloc[top_content_idx]['id'].tolist()
            content_bgg_ids = [i for i, idx in zip(content_bgg_ids, top_content_idx) if content_sims[idx] >= 0]
            
        # Unified set
        unified_ids = list(set(candidate_bgg_ids) | set(content_bgg_ids))
        
        # Filter out already rated
        final_candidates = [i for i in unified_ids if int(i) not in all_rated_bgg_ids]
        
        # Diversity Filter
        if not liked_indices.empty and final_candidates:
            temp_candidates = self.content.df[self.content.df['id'].isin(final_candidates)].copy()
            filtered_idx, _ = self.content._filter_family_games(liked_indices, temp_candidates)
            final_candidates = self.content.df.loc[filtered_idx, 'id'].tolist()

        # Phase 4: Final Scoring
        final_list = []
        for bgg_id in final_candidates:
            # SVD Score
            norm_svd = 0.0
            if is_known_to_svd:
                try:
                    g_idx = self.svd.game_enc.transform([str(bgg_id)])[0]
                    raw_svd = self.svd.predict([u_idx], [g_idx])[0]
                    norm_svd = float(np.clip((raw_svd - 1) / 9.0, 0, 1))
                except: norm_svd = 0.0
            
            # Content Score
            norm_content = 0.0
            if user_seed_emb is not None:
                idx_arr = self.content.df[self.content.df['id'] == int(bgg_id)].index
                if not idx_arr.empty:
                    norm_content = float(cosine_similarity([user_seed_emb], [self.content.embeddings[idx_arr[0]]])[0][0])

            hybrid_score = (svd_weight * norm_svd) + (content_weight * norm_content)
            
            lookup = self.content.df[self.content.df['id'] == int(bgg_id)]
            if not lookup.empty:
                row = lookup.iloc[0]
                final_list.append({
                    "BGGId": int(bgg_id),
                    "Name": row['name'],
                    "Year": str(int(row['year'])) if pd.notna(row['year']) else "N/A",
                    "Image": row['image'] if 'image' in lookup.columns else None,
                    "Description": row['description'] if 'description' in lookup.columns else "",
                    "SVD_Score": float(norm_svd),
                    "Content_Score": float(norm_content),
                    "Hybrid_Score": float(hybrid_score),
                    "HasBoth": (norm_svd > 0 and norm_content > 0)
                })

        if not final_list: return pd.DataFrame()
        hybrid_df = pd.DataFrame(final_list).sort_values(["HasBoth", "Hybrid_Score"], ascending=[False, False]).head(n)
        # self._print_hybrid_results(username, hybrid_df)
        return hybrid_df

    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Applies metadata filters to the given DataFrame."""
        if not filters:
            return df
            
        filtered_df = df.copy()
        
        # 1. Players
        if 'min_players' in filters and filters['min_players'] is not None:
            filtered_df = filtered_df[filtered_df['max_players'] >= filters['min_players']]
        if 'max_players' in filters and filters['max_players'] is not None:
            filtered_df = filtered_df[filtered_df['min_players'] <= filters['max_players']]
            
        # 2. Playtime
        if 'min_playtime' in filters and filters['min_playtime'] is not None:
            filtered_df = filtered_df[filtered_df['max_playtime'] >= filters['min_playtime']]
        if 'max_playtime' in filters and filters['max_playtime'] is not None:
            filtered_df = filtered_df[filtered_df['min_playtime'] <= filters['max_playtime']]
            
        # 3. Weight (Weight is 1-5)
        if 'min_weight' in filters and filters['min_weight'] is not None:
            filtered_df = filtered_df[filtered_df['weight'] >= filters['min_weight']]
        if 'max_weight' in filters and filters['max_weight'] is not None:
            filtered_df = filtered_df[filtered_df['weight'] <= filters['max_weight']]
            
        # 4. Year
        if 'min_year' in filters and filters['min_year'] is not None:
            filtered_df = filtered_df[filtered_df['year'] >= filters['min_year']]
        if 'max_year' in filters and filters['max_year'] is not None:
            filtered_df = filtered_df[filtered_df['year'] <= filters['max_year']]
            
        # 5. Thai Version
        if filters.get('has_thai_version'):
            filtered_df = filtered_df[filtered_df['has_thai_version'] == True]
            
        # 6. Categories (Match ANY or Match ALL)
        if 'categories' in filters and filters['categories']:
            match_mode = filters.get('category_match', 'any')
            from ast import literal_eval
            def check_cats(x):
                try:
                    target_cats = set(filters['categories'])
                    game_cats = set(literal_eval(x) if isinstance(x, str) else x)
                    if match_mode == 'all':
                        return target_cats.issubset(game_cats)
                    return not target_cats.isdisjoint(game_cats)
                except: return False
            filtered_df = filtered_df[filtered_df['categories'].apply(check_cats)]
            
        # 7. Mechanics (Match ANY if list provided)
        if 'mechanics' in filters and filters['mechanics']:
            from ast import literal_eval
            def check_mechs(x):
                try:
                    target_mechs = set(filters['mechanics'])
                    game_mechs = set(literal_eval(x) if isinstance(x, str) else x)
                    return not target_mechs.isdisjoint(game_mechs)
                except: return False
            filtered_df = filtered_df[filtered_df['mechanics'].apply(check_mechs)]
            
        return filtered_df

    def _print_hybrid_results(self, username, df):
        width = 100
        print(f"\n{Colors.PURPLE}{'='*width}{Colors.END}")
        print(f"{Colors.BOLD}HYBRID RECOMMENDATIONS FOR:{Colors.END} {Colors.GREEN}{username}{Colors.END}")
        print(f"{Colors.PURPLE}{'='*width}{Colors.END}")
        header = f"{'#':<3} | {'Game Name':<35} | {'SVD Norm':<10} | {'Content':<10} | {'Hybrid Score':<12} | {'Link'}"
        print(header)
        print("-" * width)
        for i, row in enumerate(df.itertuples(), 1):
            url = f"https://boardgamegeek.com/boardgame/{row.BGGId}"
            line = f"{i:<3} | {row.Name[:35]:<35} | {row.SVD_Score:<10.2f} | {row.Content_Score:<10.2f} | {row.Hybrid_Score:<12.2f} | {url}"
            print(line)
        print(f"{Colors.PURPLE}{'='*width}{Colors.END}\n")

    def recommend_by_seeds(self, seed_bgg_ids: List[int], n: int = 15, filters: Dict = None):
        """Pure content-based recommendations from a list of BGGIds (Public usage)."""
        print(f"\n{Colors.PURPLE}🧠 GENERATING SEED-BASED RECOMMENDATIONS FOR {len(seed_bgg_ids)} GAMES...{Colors.END}")
        
        # 1. Global Filtering (Full Pool)
        if filters:
            filtered_pool_df = self._apply_filters(self.content.df, filters)
            print(f"   {Colors.BLUE}[*]{Colors.END} Filtered pool: {len(filtered_pool_df)} / {len(self.content.df)} games match criteria.")
            if filtered_pool_df.empty:
                return pd.DataFrame()
            pool_indices = filtered_pool_df.index
        else:
            pool_indices = self.content.df.index

        # 2. Get embeddings for seed games
        seed_indices = self.content.df[self.content.df['id'].isin(seed_bgg_ids)].index
        if seed_indices.empty:
            return pd.DataFrame()
            
        seed_embeddings = self.content.embeddings[seed_indices]
        mean_seed_emb = np.mean(seed_embeddings, axis=0)
        
        # 3. Calculate similarities only for the filtered pool
        filtered_embeddings = self.content.embeddings[pool_indices]
        sims_narrow = np.clip(cosine_similarity([mean_seed_emb], filtered_embeddings).flatten(), 0, 1)
        
        # 4. Filter out the seed games themselves if they are in the pool
        for s_idx in seed_indices:
            if s_idx in pool_indices:
                # Find the position in pool_indices
                pos = np.where(pool_indices == s_idx)[0]
                if len(pos) > 0:
                    sims_narrow[pos[0]] = -1.0
        
        # 5. Get top candidates for diversity filtering
        top_n_narrow = min(len(sims_narrow), 200)
        if top_n_narrow == 0: return pd.DataFrame()
        
        top_local_idx = np.argpartition(sims_narrow, -top_n_narrow)[-top_n_narrow:]
        top_local_idx = top_local_idx[np.argsort(sims_narrow[top_local_idx])[::-1]] # Sort properly
        
        candidates_df = self.content.df.iloc[pool_indices[top_local_idx]].copy()
        candidates_df['similarity_score'] = sims_narrow[top_local_idx]
        
        # 6. DIVERSITY FILTER: Remove expansions and versions of the seed games
        filtered_indices_abs, _ = self.content._filter_family_games(seed_indices, candidates_df)
        
        # 7. Final selection
        final_df = candidates_df.loc[filtered_indices_abs].head(n)
        
        final_list = []
        for _, row in final_df.iterrows():
            final_list.append({
                "BGGId": int(row['id']),
                "Name": row['name'],
                "Year": int(row['year']) if pd.notna(row['year']) else 0,
                "Image": row['image'] if 'image' in row else None,
                "Description": row['description'] if 'description' in row else "",
                "SVD_Score": 0.0,
                "Content_Score": float(row['similarity_score']),
                "Hybrid_Score": float(row['similarity_score']), 
                "HasBoth": False
            })
            
        return pd.DataFrame(final_list)

def main():
    try:
        # Configuration
        TARGET_USER = "your_username"
        
        hybrid = HybridRecommender()
        hybrid.recommend(TARGET_USER, n=10)
        
    except Exception:
        print(f"\n{Colors.RED}[!] A critical error occurred:{Colors.END}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
