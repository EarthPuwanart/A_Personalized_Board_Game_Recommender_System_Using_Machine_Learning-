import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
import time
import warnings
import joblib
import sys
import io
import traceback
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

# --- Configuration & Environment Setup ---
def setup_environment():
    """Configures the environment for silent and correct operation on Windows."""
    if sys.platform == "win32":
        # Ensure UTF-8 output for console icons/colors
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    warnings.filterwarnings("ignore")

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

class BoardGameSVD:
    def __init__(self, n_components=50, rating_scale=(1, 10)):
        self.n_components = n_components
        self.rating_scale = rating_scale
        
        self.user_enc = LabelEncoder()
        self.game_enc = LabelEncoder()
        
        self.user_factors = None
        self.game_factors = None
        
        # Biases
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
        
        self.popularity = None
        self._sparse = None
        self._fitted = False

    def fit(self, df):
        """Train the model with User and Item Biases + SVD Interaction"""
        print(f"\n{Colors.BLUE}🚀 FITTING MODEL (k={self.n_components}){Colors.END}")
        t0 = time.time()

        # --------- Data Cleaning ---------
        df = df[["Username", "BGGId", "Rating"]].copy()
        df["BGGId"] = df["BGGId"].astype(str)
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.dropna()
        df["Rating"] = df["Rating"].clip(*self.rating_scale)

        print(f"   {Colors.BOLD}[*]{Colors.END} Ratings count: {len(df):,}")

        # --------- Encoding ---------
        self.user_enc.fit(df["Username"])
        self.game_enc.fit(df["BGGId"])
        
        df["user_idx"] = self.user_enc.transform(df["Username"])
        df["game_idx"] = self.game_enc.transform(df["BGGId"])

        n_users = len(self.user_enc.classes_)
        n_games = len(self.game_enc.classes_)

        # --------- Sparse Matrix ---------
        sparse = sp.csr_matrix(
            (
                df["Rating"].astype(np.float32).values,
                (df["user_idx"].values, df["game_idx"].values),
            ),
            shape=(n_users, n_games),
            dtype=np.float32
        )
        self._sparse = sparse 

        # --------- Calculate Biases ---------
        self.global_mean = float(df["Rating"].mean())
        print(f"   {Colors.BOLD}[*]{Colors.END} Global mean: {self.global_mean:.4f}")

        user_sums = np.asarray(sparse.sum(axis=1)).ravel()
        user_counts = np.diff(sparse.indptr)
        self.user_biases = np.divide(
            user_sums - (user_counts * self.global_mean),
            user_counts + 10, 
            out=np.zeros_like(user_sums),
            where=user_counts != 0
        )

        row_ids = np.repeat(np.arange(n_users), user_counts)
        col_ids = sparse.indices
        centered_data = sparse.data - self.global_mean - self.user_biases[row_ids]
        
        temp_sparse = sp.csr_matrix((centered_data, (row_ids, col_ids)), shape=(n_users, n_games))
        item_sums = np.asarray(temp_sparse.sum(axis=0)).ravel()
        item_counts = np.asarray((temp_sparse != 0).sum(axis=0)).ravel()
        
        self.item_biases = np.divide(
            item_sums,
            item_counts + 10,
            out=np.zeros_like(item_sums),
            where=item_counts != 0
        )
        
        self.popularity = np.asarray(sparse.mean(axis=0)).ravel()

        # --------- Matrix Centering for SVD ---------
        centered_data -= self.item_biases[col_ids]
        centered = sp.csr_matrix((centered_data, (row_ids, col_ids)), shape=(n_users, n_games))

        # --------- SVD ---------
        max_k = min(n_users - 1, n_games - 1)
        k = min(self.n_components, max_k)
        
        print(f"   {Colors.BOLD}[*]{Colors.END} Using k={k}")
        svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
        
        self.user_factors = svd.fit_transform(centered)
        self.game_factors = svd.components_.T

        explained = svd.explained_variance_ratio_.sum()
        print(f"   {Colors.BOLD}[*]{Colors.END} Explained variance: {explained:.4f} ({explained*100:.2f}%)")

        self._fitted = True
        elapsed = time.time() - t0
        print(f"{Colors.GREEN}✅ Training completed in {elapsed:.2f}s{Colors.END}")
        
        del df, centered, temp_sparse, centered_data
        gc.collect()
        return self

    def predict(self, user_indices, game_indices):
        interaction = np.sum(self.user_factors[user_indices] * self.game_factors[game_indices], axis=1)
        preds = self.global_mean + self.user_biases[user_indices] + self.item_biases[game_indices] + interaction
        return np.clip(preds, *self.rating_scale)

    def evaluate(self, df_test):
        if not self._fitted: raise RuntimeError("Model not fitted")
        print(f"\n{Colors.PURPLE}📊 EVALUATING MODEL{Colors.END}")
        
        df_test = df_test[["Username", "BGGId", "Rating"]].copy()
        df_test["BGGId"] = df_test["BGGId"].astype(str)
        mask = (df_test["Username"].isin(self.user_enc.classes_)) & (df_test["BGGId"].isin(self.game_enc.classes_))
        df_test = df_test[mask].copy()

        if len(df_test) == 0:
            print(f"{Colors.YELLOW}   [!] No overlap for evaluation{Colors.END}")
            return None

        user_idx = self.user_enc.transform(df_test["Username"])
        game_idx = self.game_enc.transform(df_test["BGGId"])
        
        preds = self.predict(user_idx, game_idx)
        actuals = df_test["Rating"].values
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        
        print(f"   {Colors.BOLD}[*]{Colors.END} Test Samples: {len(df_test):,}")
        print(f"   {Colors.GREEN}RMSE: {rmse:.4f}{Colors.END}")

        # Ranking Metrics
        threshold = 7.5; k = 10
        df_test["pred"] = preds
        df_test["is_relevant"] = df_test["Rating"] >= threshold
        
        precisions = []; recalls = []
        test_users = df_test["Username"].unique()
        

        if len(test_users) > 1000:
            test_users = np.random.choice(test_users, 1000, replace=False)
            
        print(f"   {Colors.BOLD}[*]{Colors.END} Calculating Ranking Metrics...")
        for user in tqdm(test_users, desc="      Evaluating", unit="user"):
            user_data = df_test[df_test["Username"] == user]
            n_rel = user_data["is_relevant"].sum()
            if n_rel == 0: continue
            
            top_k_preds = user_data.nlargest(k, "pred")
            n_rel_and_rec = top_k_preds["is_relevant"].sum()
            precisions.append(n_rel_and_rec / k)
            recalls.append(n_rel_and_rec / n_rel)
            
        avg_prec = np.mean(precisions) if precisions else 0.0
        avg_rec = np.mean(recalls) if recalls else 0.0
        
        if precisions:
            print(f"   {Colors.BOLD}[*]{Colors.END} Precision@{k}: {avg_prec:.4f}")
            print(f"   {Colors.BOLD}[*]{Colors.END} Recall@{k}: {avg_rec:.4f}")
            
        return {
            "RMSE": rmse,
            "Precision@10": avg_prec,
            "Recall@10": avg_rec
        }

    def recommend(self, username, n=10):
        if not self._fitted: raise RuntimeError("Model not fitted")
        
        if username in self.user_enc.classes_:
            u_idx = self.user_enc.transform([username])[0]
            interaction = self.game_factors @ self.user_factors[u_idx]
            scores = self.global_mean + self.user_biases[u_idx] + self.item_biases + interaction
            rated = self._sparse.getrow(u_idx).indices
            scores[rated] = -1.0
            method = "SVD (User-Specific)"
        else:
            scores = self.popularity.copy()
            method = f"Popularity (Global - Cold Start)"
            
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        
        res = pd.DataFrame({
            "BGGId": self.game_enc.inverse_transform(top_idx),
            "PredictedRating": np.round(scores[top_idx], 4),
            "Method": method
        })
        
        # Pretty Print Results
        width = 80
        print(f"\n{Colors.PURPLE}{'='*width}{Colors.END}")
        print(f"{Colors.BOLD}TOP RECOMMENDATIONS FOR:{Colors.END} {Colors.GREEN}{username}{Colors.END}")
        print(f"{Colors.YELLOW}Method: {method}{Colors.END}")
        print(f"{Colors.PURPLE}{'='*width}{Colors.END}")
        header = f"{'#':<3} | {'BGGId':<10} | {'Pred. Rating':<13} | {'Link':<40}"
        print(header)
        print("-" * width)
        for i, row in enumerate(res.itertuples(), 1):
            url = f"https://boardgamegeek.com/boardgame/{row.BGGId}"
            print(f"{i:<3} | {str(row.BGGId):<10} | {row.PredictedRating:<13.4f} | {url:<40}")
        print(f"{Colors.PURPLE}{'='*width}{Colors.END}\n")
        
        return res

    def get_user_history(self, username, threshold=8.0):
        """Returns a list of BGGIds (and ratings) for games the user rated highly."""
        if username not in self.user_enc.classes_:
            return []
            
        u_idx = self.user_enc.transform([username])[0]
        user_row = self._sparse.getrow(u_idx)
        
        # indices of games rated
        game_indices = user_row.indices
        ratings = user_row.data
        
        # filter by threshold
        mask = ratings >= threshold
        liked_indices = game_indices[mask]
        liked_ratings = ratings[mask]
        
        # Sort by rating descending
        sort_idx = np.argsort(liked_ratings)[::-1]
        
        results = []
        for idx in liked_indices[sort_idx]:
            results.append(self.game_enc.inverse_transform([idx])[0])
            
        return results

    def save(self, filename):
        tmp_filename = filename + ".tmp"
        print(f"💾 {Colors.BLUE}Saving model to {tmp_filename}...{Colors.END}")
        joblib.dump(self, tmp_filename)
        # Atomic rename prevents truncated file reads by concurrent requests
        os.replace(tmp_filename, filename)
        print(f"{Colors.GREEN}✅ Saved to {filename}!{Colors.END}")

    @staticmethod
    def load(filename):
        print(f"📂 {Colors.BLUE}Loading model from {filename}...{Colors.END}")
        model = joblib.load(filename)
        print(f"{Colors.GREEN}✅ Loaded!{Colors.END}")
        return model

def train_svd_model(quick_mode=True, k_factors=50, force=True):
    """
    Standalone training function to be called from API or CLI.
    quick_mode=True: Uses 100% of data, skips evaluation.
    quick_mode=False: Uses 80/20 split, performs evaluation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "svd_model.joblib")
    data_path = os.path.join(script_dir, "data", "user_ratings.csv")

    if not os.path.exists(data_path):
        print(f"{Colors.RED}[!] Data file not found at {data_path}{Colors.END}")
        return False

    try:
        print(f"\n{Colors.PURPLE}--- SVD TRAINING (Mode: {'QUICK' if quick_mode else 'FULL'}) ---{Colors.END}")
        
        # 1. Load Main Data
        df = pd.read_csv(data_path, usecols=["Username", "BGGId", "Rating"])
        
        # 2. Merge New Ratings
        real_ratings_path = os.path.join(script_dir, "..", "real_user_ratings.csv")
        if os.path.exists(real_ratings_path):
            print(f"➕ {Colors.GREEN}Merging new ratings from real_user_ratings.csv...{Colors.END}")
            real_df = pd.read_csv(real_ratings_path, usecols=["Username", "BGGId", "Rating"])
            df = pd.concat([df, real_df], ignore_index=True)
            print(f"   {Colors.BOLD}[*]{Colors.END} Total ratings for training: {len(df):,}")

        model = BoardGameSVD(n_components=k_factors)
        
        if quick_mode:
            print(f"⚡ {Colors.BLUE}Quick Train: Using 100% data, skipping evaluation...{Colors.END}")
            model.fit(df)
        else:
            print(f"📊 {Colors.BLUE}Full Train: Using 80/20 split and evaluating...{Colors.END}")
            rng = np.random.default_rng(42)
            mask = rng.random(len(df)) < 0.8
            df_train, df_test = df[mask].copy(), df[~mask].copy()
            model.fit(df_train)
            model.evaluate(df_test)

        # 3. Save Model
        model.save(model_path)
        print(f"{Colors.GREEN}✅ Model saved to {model_path}{Colors.END}")
        return True

    except Exception:
        print(f"\n{Colors.RED}[!] Training failed:{Colors.END}")
        traceback.print_exc()
        return False

# --- Main Execution ---
def main():
    # ==========================================
    # ⚙️ USER SETTINGS (Easy configuration)
    # ==========================================
    QUICK_MODE = False        # Set to True for fast training (no RMSE evaluation)
    K_FACTORS = 50           # Number of latent factors
    # ==========================================

    train_svd_model(quick_mode=QUICK_MODE, k_factors=K_FACTORS)

if __name__ == "__main__":
    main()
