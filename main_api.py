from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Body, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uvicorn
import os
import sys
import time
import csv
import json
import random
import pandas as pd
import asyncio
import difflib
from typing import Optional, Any

# Ensure we can import from the subdirectories
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'collaborative'))
sys.path.append(os.path.join(script_dir, 'content-based'))

try:
    from svd_recommender import BoardGameSVD, Colors, train_svd_model
    from hybrid_recommender import HybridRecommender
except ImportError:
    # Manual path fix if run from different dir or environment issues
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from svd_recommender import BoardGameSVD, Colors, train_svd_model
    from hybrid_recommender import HybridRecommender

# Essential for joblib to find the class if the model was saved from a script where BoardGameSVD was in __main__
try:
    import sys
    if 'BoardGameSVD' not in dir(sys.modules['__main__']):
        sys.modules['__main__'].BoardGameSVD = BoardGameSVD
except (AttributeError, KeyError):
    pass

# 1. Lifespan for Model Loading (Modern FastAPI way)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(f"{Colors.PURPLE}--- STARTING API SERVER ---{Colors.END}")
    print(f"📖 {Colors.BLUE}Initializing Recommender Models...{Colors.END}")
    start_time = time.time()
    try:
        models["recommender"] = HybridRecommender()
        elapsed = time.time() - start_time
        print(f"{Colors.GREEN}✅ Models loaded successfully in {elapsed:.2f}s{Colors.END}")
        print(f"📁 {Colors.BLUE}User Database:{Colors.END} {USER_DB_PATH}")
        print(f"📁 {Colors.BLUE}Ratings Database:{Colors.END} {os.path.join(script_dir, 'real_user_ratings.csv')}")
    except Exception as e:
        print(f"{Colors.RED}[!] Initialization failed: {e}{Colors.END}")
        models["recommender"] = None
    # Cache names for fuzzy search
    if models["recommender"]:
        models['all_names'] = models["recommender"].content.df['name'].astype(str).tolist()
        print(f"✅ {Colors.GREEN}Search cache ready with {len(models['all_names'])} games.{Colors.END}")
    
    yield
    # --- Shutdown ---
    print(f"{Colors.YELLOW}--- SHUTTING DOWN API SERVER ---{Colors.END}")
    models.clear()

# 2. Initialize FastAPI
app = FastAPI(
    title="Board Game Recommender API",
    lifespan=lifespan
)

# 3. Add CORS Support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Auth & Security Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
USER_DB_PATH = os.path.join(script_dir, "users_db.json")
RATINGS_CSV_PATH = os.path.join(script_dir, "real_user_ratings.csv")

# 5. Data Models
class UserRating(BaseModel):
    username: str
    game_id: int
    rating: float

class UserAuth(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class SeedRequest(BaseModel):
    seed_ids: list[int]
    filters: Optional[dict[str, Any]] = None

# 6. Helper Functions
def get_users_db():
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'w') as f:
            json.dump({}, f)
    with open(USER_DB_PATH, 'r') as f:
        return json.load(f)

def save_users_db(db):
    with open(USER_DB_PATH, 'w') as f:
        json.dump(db, f, indent=4)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 7. Endpoints
@app.get("/")
def read_root():
    return FileResponse(os.path.join(script_dir, "index.html"))

@app.post("/register")
def register(user: UserAuth):
    db = get_users_db()
    if user.username in db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = pwd_context.hash(user.password)
    db[user.username] = {"password": hashed_password}
    save_users_db(db)
    
    print(f"👤 {Colors.GREEN}New User Registered:{Colors.END} {user.username}")
    return {"message": "User registered successfully"}

@app.post("/login")
def login(user: UserAuth):
    db = get_users_db()
    db_user = db.get(user.username)
    
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    print(f"🔑 {Colors.BLUE}User Logged In:{Colors.END} {user.username}")
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

# --- Background Tasks ---
# Use an asyncio lock to prevent multiple concurrent training jobs
training_lock = asyncio.Lock()

async def task_retrain_and_reload():
    """Retrains the SVD model in Quick Mode and reloads it into the recommender."""
    if training_lock.locked():
        print("⏳ [Background] Training already in progress, skipping duplicate request.")
        return
        
    async with training_lock:
        # Run the CPU-intensive training in a thread to unblock the async event loop
        success = await asyncio.to_thread(train_svd_model, quick_mode=True)
        recommender = models.get("recommender")
        if success and recommender:
            recommender.reload_svd()
            print("🚀 [Background] SVD Model Retrained and Reloaded successfully!")

@app.post("/train-svd")
async def manual_train(background_tasks: BackgroundTasks):
    """Manually trigger a background retraining."""
    background_tasks.add_task(task_retrain_and_reload)
    return {"message": "Retraining started in background"}

@app.post("/rate")
def save_rating(data: UserRating, background_tasks: BackgroundTasks):
    """Saves a new user rating and triggers background retraining."""
    res = save_to_csv(data.username, data.game_id, data.rating)
    background_tasks.add_task(task_retrain_and_reload)
    return res

@app.post("/rate-bulk")
def save_ratings_bulk(background_tasks: BackgroundTasks, username: str = Body(...), ratings: list = Body(...)):
    """Saves multiple ratings at once and triggers background retraining."""
    file_path = RATINGS_CSV_PATH
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            new_ratings_dict = {item['game_id']: item['rating'] for item in ratings}
            
            # Update existing ratings
            user_mask = df['Username'].str.lower() == username.lower()
            for idx in df[user_mask].index:
                game_id = df.loc[idx, 'BGGId']
                if game_id in new_ratings_dict:
                    df.loc[idx, 'Rating'] = new_ratings_dict[game_id]
                    df.loc[idx, 'Timestamp'] = timestamp
                    del new_ratings_dict[game_id]  # Remove so we know what's left to append
            
            # Append remaining new ratings
            if new_ratings_dict:
                new_rows = [{"Username": username, "BGGId": gid, "Rating": r, "Timestamp": timestamp} for gid, r in new_ratings_dict.items()]
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            
            df.to_csv(file_path, index=False)
        else:
            new_rows = [{"Username": username, "BGGId": item['game_id'], "Rating": item['rating'], "Timestamp": timestamp} for item in ratings]
            df = pd.DataFrame(new_rows)
            df.to_csv(file_path, index=False)
        
        # Trigger background retraining automatically!
        background_tasks.add_task(task_retrain_and_reload)
        return {"status": "success", "message": f"{len(ratings)} ratings saved and optimization started for {username}"}
    except Exception as e:
        print(f"Error saving bulk to CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def save_to_csv(username, game_id, rating):
    file_path = RATINGS_CSV_PATH
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Check if this user already rated this game
            mask = (df['Username'].str.lower() == username.lower()) & (df['BGGId'] == game_id)
            if not df[mask].empty:
                # Update existing rating
                df.loc[mask, 'Rating'] = rating
                df.loc[mask, 'Timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Append new rating
                new_row = pd.DataFrame([{"Username": username, "BGGId": game_id, "Rating": rating, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(file_path, index=False)
        else:
            # Create new file with one rating
            df = pd.DataFrame([{"Username": username, "BGGId": game_id, "Rating": rating, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}])
            df.to_csv(file_path, index=False)
        return {"status": "success", "message": "Rating saved/updated"}
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-ratings/{username}")
def get_user_ratings(username: str):
    """Fetch all ratings for a user with game names."""
    if not os.path.exists(RATINGS_CSV_PATH):
        return []
    
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        df_ratings = pd.read_csv(RATINGS_CSV_PATH)
        user_df = df_ratings[df_ratings['Username'].str.lower() == username.lower()]
        
        if user_df.empty:
            return []

        # Join with content dataframe to get names and images
        # Optimization: Only extract records for the games the user rated, instead of copying all 100K+ rows
        rated_bgg_ids = user_df['BGGId'].astype(int).tolist()
        content_df = recommender.content.df[recommender.content.df['id'].isin(rated_bgg_ids)][['id', 'name', 'image', 'year']].copy()
        
        # Convert user_df['BGGId'] and content_df['id'] to same type for merge
        user_df = user_df.copy()
        user_df.loc[:, 'BGGId'] = user_df['BGGId'].astype(int)
        content_df.loc[:, 'id'] = content_df['id'].astype(int)
        
        merged = pd.merge(user_df, content_df, left_on='BGGId', right_on='id')
        
        result = []
        for _, row in merged.iterrows():
            result.append({
                "BGGId": int(row['BGGId']),
                "Name": row['name'],
                "Rating": float(row['Rating']),
                "Image": row['image'],
                "Year": int(row['year']),
                "Timestamp": row['Timestamp']
            })
        # Sort by timestamp descending
        result.sort(key=lambda x: str(x['Timestamp']), reverse=True)
        return result
    except Exception as e:
        print(f"Error fetching user ratings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user-ratings/{username}/{game_id}")
def delete_rating(username: str, game_id: int, background_tasks: BackgroundTasks):
    """Delete a specific rating and trigger retraining."""
    if not os.path.exists(RATINGS_CSV_PATH):
        raise HTTPException(status_code=404, detail="No ratings found")
    
    try:
        df = pd.read_csv(RATINGS_CSV_PATH)
        mask = (df['Username'].str.lower() == username.lower()) & (df['BGGId'] == game_id)
        if df[mask].empty:
            raise HTTPException(status_code=404, detail="Rating not found")
        
        df = df[~mask]
        df.to_csv(RATINGS_CSV_PATH, index=False)
        
        # Trigger retraining
        background_tasks.add_task(task_retrain_and_reload)
        return {"status": "success", "message": "Rating deleted"}
    except Exception as e:
        print(f"Error deleting rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popular-games")
def get_popular_games(n: int = 30, thai: bool = False):
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Sort by users_rated descending and take top 500, then sample n
    try:
        df = recommender.content.df
        if thai:
            df = df[df['has_thai_version'] == True]
            
        # Ensure we don't try to sample more than available
        available_count = len(df.head(500))
        sample_n = min(n, available_count)
        
        if sample_n == 0:
            return []
            
        popular = df.head(500).sample(sample_n)
        result = []
        for _, row in popular.iterrows():
            result.append({
                "BGGId": int(row['id']),
                "Name": row['name'],
                "Year": int(row['year']),
                "Image": row['image'],
                "Description": row['description']
            })
        return result
    except Exception as e:
        print(f"Popular Games Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/discover-feed")
def get_discover_feed():
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        df = recommender.content.df
        rows = []
        hero_game = None
        
        # Helper to format games
        def format_games(games_df):
            result = []
            for _, row in games_df.iterrows():
                result.append({
                    "BGGId": int(row['id']),
                    "Name": str(row['name']),
                    "Year": int(row['year']) if pd.notna(row['year']) else 0,
                    "Image": str(row['image']),
                    "Description": str(row['description'])
                })
            return result

        # 1. Available in Thai (Select Hero first)
        try:
            thai_games_all = df[df['has_thai_version'] == True]
            if not thai_games_all.empty:
                # Pick 1 for Hero
                hero_row = thai_games_all.sample(1).iloc[0]
                hero_game = {
                    "BGGId": int(hero_row['id']),
                    "Name": str(hero_row['name']),
                    "Year": int(hero_row['year']) if pd.notna(hero_row['year']) else 0,
                    "Image": str(hero_row['image']),
                    "Description": str(hero_row['description'])
                }
                
                # Filter out the hero from the row
                thai_games_for_row = thai_games_all[thai_games_all['id'] != hero_game['BGGId']]
                rows.append({"id": "row-thai", "title": "Available in Thai 🇹🇭", "games": format_games(thai_games_for_row.sample(min(30, len(thai_games_for_row))))})
        except Exception as e:
            print(f"Error Thai/Hero: {e}")

        # 2. Fresh Arrivals
        try:
            max_year = df['year'].max()
            new_games = df[df['year'] >= (max_year - 2)]
            if not new_games.empty:
                rows.append({"id": "row-new", "title": "Fresh Arrivals ✨", "games": format_games(new_games.sample(min(50, len(new_games))))})
        except Exception as e:
            print(f"Error New: {e}")
            
        # 3. Trending Now
        try:
            hot_path = os.path.join(script_dir, 'content-based', 'data', 'hot_boardgames.csv')
            if os.path.exists(hot_path):
                hot_df = pd.read_csv(hot_path)
                rows.append({"id": "row-trending", "title": "Trending Now 🔥", "games": format_games(hot_df.sample(min(50, len(hot_df))))})
        except Exception as e:
            print(f"Error Trending: {e}")

        # 4. All-Time Legends
        try:
            legends = df.sort_values(by=['users_rated', 'bayes_rating'], ascending=False).head(200)
            rows.append({"id": "row-legends", "title": "All-Time Legends 👑", "games": format_games(legends.sample(min(50, len(legends))))})
        except Exception as e:
            print(f"Error Legends: {e}")

        # Shuffle the first 4 rows so they appear in a different order each time
        random.shuffle(rows)

        # 5-7. Random 3 Filters
        try:
            filters = [
                {"id": "row-heavy", "title": "Brain Burners 🧠", "df": df[df['weight'] >= 3]},
                {"id": "row-light", "title": "Easy to Learn 🎈", "df": df[df['weight'] < 3]},
                {"id": "row-quick", "title": "Quick Plays ⚡", "df": df[df['max_playtime'] < 30]},
                {"id": "row-standard", "title": "Mid-Length ⏱️", "df": df[(df['max_playtime'] >= 30) & (df['max_playtime'] < 90)]},
                {"id": "row-epic", "title": "The Long Haul ⏳", "df": df[df['min_playtime'] >= 90]},
                {"id": "row-solo", "title": "Solo or Duo 🎲", "df": df[df['min_players'] <= 2]},
                {"id": "row-group", "title": "Small Group 🍕", "df": df[(df['min_players'] >= 3) & (df['min_players'] <= 6)]},
                {"id": "row-party", "title": "Party Time 🎉", "df": df[df['max_players'] >= 7]}
            ]

            valid_filters = [f for f in filters if len(f["df"]) >= 50]
            chosen_filters = random.sample(valid_filters, min(3, len(valid_filters)))
            for f in chosen_filters:
                rows.append({"id": f["id"], "title": f["title"], "games": format_games(f["df"].sample(50))})
        except Exception as e:
            print(f"Error Filters: {e}")
            
        # 8-15. Random 8 Themes from Top Pool
        try:
            pool_path = os.path.join(script_dir, 'theme_analysis_with_elbow.csv')
            if os.path.exists(pool_path):
                theme_df = pd.read_csv(pool_path)
                # Take top 150 themes as the pool
                top_pool = theme_df.head(150)
                
                # Exclude meta-tags or too specific ones if desired
                exclude = ['Kickstarter', 'Better Description Needed!', 'Upcoming Releases', 'Wargame', 'Tabletopia', 'VASSAL', 'TableTop Simulator Mod (TTS)']
                top_pool = top_pool[~top_pool['Theme'].isin(exclude)]
                
                # Further refine for shorter/cleaner titles (e.g. < 40 chars)
                top_pool = top_pool[top_pool['Theme'].str.len() < 40]
                
                # Sample 8 random themes from the pool
                sample_count = min(8, len(top_pool))
                chosen_themes = top_pool.sample(sample_count)['Theme'].tolist()
                
                added_themes = 0
                for theme_name in chosen_themes:
                    if added_themes >= 8: break
                    
                    # Search across all columns
                    mask = (df['categories'].str.contains(theme_name, case=False, na=False) | 
                            df['mechanics'].str.contains(theme_name, case=False, na=False) | 
                            (df['families'].str.contains(theme_name, case=False, na=False) if 'families' in df.columns else False))
                    
                    theme_games = df[mask]
                    if len(theme_games) >= 50:
                        rows.append({
                            "id": f"row-dynamic-{added_themes}", 
                            "title": theme_name, 
                            "games": format_games(theme_games.sample(50))
                        })
                        added_themes += 1
            else:
                # Fallback to a few common ones if CSV is missing
                fallback = ['Fantasy', 'Science Fiction', 'Card Game', 'Economic', 'Adventure']
                for i, theme_name in enumerate(fallback[:4]):
                    mask = df['categories'].str.contains(theme_name, case=False, na=False)
                    theme_games = df[mask]
                    if len(theme_games) >= 50:
                        rows.append({
                            "id": f"row-fb-{i}", 
                            "title": theme_name, 
                            "games": format_games(theme_games.sample(50))
                        })
        except Exception as e:
            print(f"Error Dynamic Themes: {e}")

        return {
            "hero": hero_game,
            "rows": rows
        }
        
    except Exception as e:
        print(f"Discover Feed Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game/{game_id}")
def get_game_details(game_id: int):
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        df = recommender.content.df
        game = df[df['id'] == game_id]
        if game.empty:
            raise HTTPException(status_code=404, detail="Game not found")
        
        row = game.iloc[0]
        return {
            "BGGId": int(row['id']),
            "Name": str(row['name']),
            "Year": int(row['year']) if pd.notna(row['year']) else 0,
            "Image": str(row['image']),
            "Description": str(row['description']),
            "MinPlayers": int(row['min_players']) if pd.notna(row['min_players']) else 0,
            "MaxPlayers": int(row['max_players']) if pd.notna(row['max_players']) else 0,
            "MinPlaytime": int(row['min_playtime']) if pd.notna(row['min_playtime']) else 0,
            "MaxPlaytime": int(row['max_playtime']) if pd.notna(row['max_playtime']) else 0,
            "Weight": float(row['weight']) if pd.notna(row['weight']) else 0,
            "Rating": float(row['bayes_rating']) if pd.notna(row['bayes_rating']) else 0,
            "Categories": str(row['categories']),
            "Mechanics": str(row['mechanics']),
            "Families": str(row['families']) if 'families' in row else "",
            "HasThai": bool(row['has_thai_version']) if 'has_thai_version' in row else False
        }
    except Exception as e:
        print(f"Get Game Detail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-games")
def search_games(
    query: str = Query(None, min_length=1), 
    exclude_user: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=100),
    min_players: int = Query(None),
    max_players: int = Query(None),
    min_playtime: int = Query(None),
    max_playtime: int = Query(None),
    min_weight: float = Query(None),
    max_weight: float = Query(None),
    min_year: int = Query(None),
    max_year: int = Query(None),
    has_thai_version: bool = Query(False),
    categories: list[str] = Query(None),
    category_match: str = Query("any", regex="^(any|all)$"),
    mechanics: list[str] = Query(None)
):
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not query:
        return {"games": [], "total": 0}

    try:
        query_lower = query.lower()
        
        filters = {
            "min_players": min_players,
            "max_players": max_players,
            "min_playtime": min_playtime,
            "max_playtime": max_playtime,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "min_year": min_year,
            "max_year": max_year,
            "has_thai_version": has_thai_version,
            "categories": categories,
            "category_match": category_match,
            "mechanics": mechanics
        }
        
        filtered_df = recommender._apply_filters(recommender.content.df, filters)
        
        # Find all matches
        all_matches = filtered_df[filtered_df['name'].str.contains(query, case=False, na=False)]
        
        # Filter out already rated games
        if exclude_user and os.path.exists(RATINGS_CSV_PATH):
            user_ratings = pd.read_csv(RATINGS_CSV_PATH)
            user_rated_ids = user_ratings[user_ratings['Username'] == exclude_user]['BGGId'].astype(int).tolist()
            # We no longer filter them out of the list; the frontend will handle badging.
        
        # Group 1: Starts with (includes exact match)
        query_df = all_matches.copy() # subset already containing query in name
        starts_with = query_df[query_df['name'].str.lower().str.startswith(query_lower)]
        all_ids = set(starts_with['id'].tolist())
        
        # Group 2: Contains in Name (and not in above)
        contains_name = query_df[~query_df['id'].isin(all_ids)]
        all_ids.update(contains_name['id'].tolist())
        
        # Group 3: Fuzzy Match (typos/similar characters)
        fuzzy_match = pd.DataFrame()
        all_names = models.get('all_names', [])
        if all_names:
            # We use lower case for better match results
            close_names = difflib.get_close_matches(query_lower, [n.lower() for n in all_names], n=40, cutoff=0.7)
            if close_names:
                fuzzy_match = filtered_df[
                    (filtered_df['name'].str.lower().isin(close_names)) & 
                    (~filtered_df['id'].isin(all_ids))
                ]
                all_ids.update(fuzzy_match['id'].tolist())
        
        # Group 4: Description Match (Keywords in description)
        desc_match = pd.DataFrame()
        if 'description' in filtered_df.columns:
            desc_match = filtered_df[
                (filtered_df['description'].str.contains(query, case=False, na=False)) & 
                (~filtered_df['id'].isin(all_ids))
            ]
        
        # Helper to sort each group by popularity (users_rated) and Quality (bayes_rating)
        def sort_group(df_group):
            if df_group is None or df_group.empty:
                return pd.DataFrame()
            # Priority: Most rated first, then highest bayes rating
            sort_cols = [c for c in ['users_rated', 'bayes_rating'] if c in df_group.columns]
            if sort_cols:
                return df_group.sort_values(by=sort_cols, ascending=False)
            return df_group

        starts_with = sort_group(starts_with)
        contains_name = sort_group(contains_name)
        fuzzy_match = sort_group(fuzzy_match)
        desc_match = sort_group(desc_match)
        
        # Combine all sorted results in order of relevance priority
        sorted_all = pd.concat([starts_with, contains_name, fuzzy_match, desc_match])
        total = len(sorted_all)
        
        # Paginate
        page = sorted_all.iloc[offset:offset + limit]
        
        result = []
        for _, row in page.iterrows():
            result.append({
                "BGGId": int(row['id']),
                "Name": str(row['name']) if pd.notna(row['name']) else "Unknown Game",
                "Year": int(row['year']) if pd.notna(row['year']) else 0,
                "Image": str(row['image']) if pd.notna(row['image']) else "",
                "Description": str(row['description']) if pd.notna(row['description']) else ""
            })
        return {"games": result, "total": total}
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-ratings-count/{username}")
def get_user_ratings_count(username: str):
    file_path = RATINGS_CSV_PATH
    if not os.path.exists(file_path):
        return {"username": username, "count": 0}
    
    try:
        count = 0
        target_user = username.lower()
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Username') and row['Username'].lower() == target_user:
                    count += 1
        return {"username": username, "count": count}
    except Exception as e:
        print(f"Error counting user ratings: {e}")
        return {"username": username, "count": 0}

@app.get("/user-status/{username}")
def get_user_status(username: str):
    """Check if the user has already submitted ratings (case-insensitive)."""
    try:
        if os.path.exists(RATINGS_CSV_PATH):
            target_user = username.lower()
            with open(RATINGS_CSV_PATH, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Username') and row['Username'].lower() == target_user:
                        return {"username": username, "has_ratings": True}
        return {"username": username, "has_ratings": False}
    except Exception as e:
        print(f"User Status Error: {e}")
        return {"username": username, "has_ratings": False}

@app.get("/recommend/{username}")
def get_recommendations(
    username: str, 
    n: int = Query(10, gt=0, le=100), 
    mode: str = Query("hybrid", regex="^(hybrid|svd|content)$"),
    min_players: int = Query(None),
    max_players: int = Query(None),
    min_playtime: int = Query(None),
    max_playtime: int = Query(None),
    min_weight: float = Query(None),
    max_weight: float = Query(None),
    min_year: int = Query(None),
    max_year: int = Query(None),
    has_thai_version: bool = Query(False),
    categories: list[str] = Query(None),
    category_match: str = Query("any", regex="^(any|all)$"),
    mechanics: list[str] = Query(None)
):
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to initialize.")
    
    # Map modes to weights
    weights = {
        "hybrid": (0.5, 0.5),
        "svd": (1.0, 0.0),
        "content": (0.0, 1.0)
    }
    svd_w, content_w = weights.get(mode, (0.5, 0.5))

    # Construct filters dictionary
    filters = {
        "min_players": min_players,
        "max_players": max_players,
        "min_playtime": min_playtime,
        "max_playtime": max_playtime,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "min_year": min_year,
        "max_year": max_year,
        "has_thai_version": has_thai_version,
        "categories": categories,
        "category_match": category_match,
        "mechanics": mechanics
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    try:
        print(f"🔍 {Colors.BOLD}API Request:{Colors.END} Recommending for {Colors.GREEN}{username}{Colors.END} (n={n}, mode={mode}, filters={len(filters)})")
        results_df = recommender.recommend(username, n=n, svd_weight=svd_w, content_weight=content_w, filters=filters)
        
        if results_df is None or results_df.empty:
            return {"username": username, "mode": mode, "recommendations": [], "count": 0}
            
        return {
            "username": username,
            "mode": mode,
            "count": len(results_df),
            "recommendations": results_df.to_dict(orient="records")
        }
    except Exception as e:
        print(f"{Colors.RED}[!] API Error: {e}{Colors.END}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/seed")
def get_seed_recommendations(request: SeedRequest, n: int = Query(15, gt=0, le=100)):
    recommender = models.get("recommender")
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    
    try:
        results_df = recommender.recommend_by_seeds(request.seed_ids, n=n, filters=request.filters)
        if results_df.empty:
            return {"recommendations": [], "count": 0}
            
        return {
            "count": len(results_df),
            "recommendations": results_df.to_dict(orient="records")
        }
    except Exception as e:
        print(f"Seed Recommendation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
