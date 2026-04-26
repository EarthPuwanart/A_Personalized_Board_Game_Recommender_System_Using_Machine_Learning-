# Personalized Board Game Recommender System

A sophisticated hybrid recommendation engine designed to provide tailored board game suggestions by combining semantic content analysis and collaborative user behavior patterns.

## 🚀 Overview

This project implements a **Hybrid Recommender System** that assists users in navigating the vast world of board games. By leveraging modern Machine Learning techniques, it addresses common issues like the "Cold Start" problem and provides a diverse range of recommendations that align with individual player preferences.

## ✨ Key Features

*   **Discover Mode**: Discover new games through various categories such as **All-Time Legends**, **Trending Now**, and **Available in Thai**, as well as specific game tags.<br><br>
<img width="1897" height="940" alt="discover_mode" src="https://github.com/user-attachments/assets/ca872bbb-a09e-4faa-919b-1fa5160628f5" /><br>

*   **Mix & Match Mode**: Receive recommendations based on the similarity of up to 5 selected games, powered primarily by the **Content-Based** filtering model.<br><br>
<img width="1896" height="942" alt="mix_match_mode" src="https://github.com/user-attachments/assets/236f0c7b-e993-44a3-8135-bbef89de900e" /><br>

*   **AI Picks Mode**: Get personalized recommendations based on your unique rating history (Login required), with the flexibility to choose from **three different recommendation models**.<br><br>
<img width="1894" height="942" alt="ai_picks_mode" src="https://github.com/user-attachments/assets/ce266981-8134-4853-ab44-abceb79d2767" /><br>

*   **User Authentication (Login)**: Secure access for users to save their preferences, managing their unique profile and rating history.<br><br>
<img width="384" height="487" alt="login_page" src="https://github.com/user-attachments/assets/cde9e45e-111f-4a4a-bd5f-f7dcf9a14e00" /><br>

*   **Manage Ratings**: A user-friendly interface to search and rate board games (1-10 scale), providing the essential data for the AI to learn and refine your personalized recommendations.<br><br>
<img width="694" height="716" alt="manage_ratings_page" src="https://github.com/user-attachments/assets/2b8b41de-c3b9-4fb8-affa-4a9c0d827cc7" /><br>

## ⚙️ Setup & Data

To run this project locally, you will need to acquire the model files and datasets. **Note: All `.csv`, `.json`, and model files have been excluded from this repository for privacy and storage reasons.**

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    Create a `.env` file or set these environment variables:
    ```env
    BGG_API_TOKEN=your_bgg_api_token_here
    JWT_SECRET_KEY=your_random_secret_key_here
    ```
4.  **Data Requirements**:
    You must provide your own data files and place them in the following structure:
    *   `users_db.json` (User authentication data)
    *   `real_user_ratings.csv`
    *   `theme_analysis_with_elbow.csv`
    *   `collaborative/svd_model.joblib`
    *   `collaborative/data/user_ratings.csv`
    *   `content-based/data/bgg_games_cleaned.csv`
    *   `content-based/data/hot_boardgames.csv`
    *   `content-based/data/bgg_games_cleaned_BAAI_bge-base-en-v1.5_embeddings.npy`
5.  **Run the application**:
    ```bash
    python main_api.py
    ```

## 🛠️ Technical Architecture

The system utilizes a dual-model approach to ensure high-quality recommendations:

1.  **Content-Based Filtering**:
    *   **Semantic Analysis**: Employs the **BGE (bge-base-en-v1.5)** language model to generate high-dimensional embeddings of board game descriptions.
    *   **Feature Engineering**: Integrates numerical metadata (play time, complexity, etc.) using `StandardScaler` for normalized similarity calculations.
    *   **Similarity**: Uses **Cosine Similarity** to find games with similar themes and mechanics.

2.  **Collaborative Filtering**:
    *   **Matrix Factorization**: Implements **Singular Value Decomposition (SVD)** to discover latent factors in user-item rating matrices.
    *   **User Patterns**: Learns from a dataset of millions of ratings to predict how a user might rate a game they haven't played yet.

3.  **Hybrid Integration**:
    *   Combines scores from both models using a weighted approach to balance accuracy (Collaborative) and discovery (Content-based).


## 📊 Evaluation Results

The system was rigorously tested using standard recommendation metrics:
*   **Precision@10** & **Recall@10**
*   **NDCG** (Normalized Discounted Cumulative Gain)
*   **MRR** (Mean Reciprocal Rank)

## 💻 Tech Stack

*   **Backend**: Python, FastAPI
*   **ML Libraries**: Scikit-learn, Surprise (SVD), Transformers (BGE model)
*   **Database/Storage**: XML (BGG API), CSV (Kaggle Dataset)
*   **Frontend**: HTML5, Vanilla CSS3, JavaScript

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
*Developed as part of a Senior Project at the Data Science Department, Faculty of Science, Silpakorn University.*
