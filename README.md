# Personalized Board Game Recommender System

A sophisticated hybrid recommendation engine designed to provide tailored board game suggestions by combining semantic content analysis and collaborative user behavior patterns.

## 🚀 Overview

This project implements a **Hybrid Recommender System** that assists users in navigating the vast world of board games. By leveraging modern Machine Learning techniques, it addresses common issues like the "Cold Start" problem and provides a diverse range of recommendations that align with individual player preferences.

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

## ✨ Key Features

*   **Discover Mode**: Discover new games through various categories such as **All-Time Legends**, **Trending Now**, and **Available in Thai**, as well as specific game tags.
*   **Mix & Match Mode**: Receive recommendations based on the similarity of up to 5 selected games, powered primarily by the **Content-Based** filtering model.
*   **AI Picks Mode**: Get personalized recommendations based on your unique rating history (Login required), with the flexibility to choose from **three different recommendation models**.

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

---
*Developed as part of a Senior Project at the Data Science Department, Faculty of Science, Silpakorn University.*
