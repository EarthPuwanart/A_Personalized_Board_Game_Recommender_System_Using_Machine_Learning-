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

## 📊 Data Source & Preprocessing

The system's intelligence is built upon a vast dataset combining official metadata and extensive community feedback:

*   **BoardGameGeek (BGG) API**: Used to fetch detailed metadata for over **135,000 board games**, including descriptions, categories, mechanics, and complexity levels.
*   **Kaggle (BGG-derived) Dataset**: Provides a massive scale of community interaction data originally sourced from BGG, featuring **18.9 million ratings** from over **411,000 users** across **21,900+ games**.
*   **Data Cleaning & Engineering**:
    *   Implemented **Winsorization** to handle outliers in playtime and player counts.
    *   Missing values were handled using **Median Imputation** for quantitative features.
    *   Textual data was synthesized into a **Semantic Context** for deep learning embedding generation.

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
