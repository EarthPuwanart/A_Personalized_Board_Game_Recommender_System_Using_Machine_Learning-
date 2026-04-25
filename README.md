# Personalized Board Game Recommender System

A sophisticated hybrid recommendation engine designed to provide tailored board game suggestions by combining semantic content analysis and collaborative user behavior patterns.

## 🚀 Overview
This project implements a **Hybrid Recommender System** that assists users in navigating the vast world of board games. By leveraging modern Machine Learning techniques, it addresses common issues like the "Cold Start" problem and provides a diverse range of recommendations that align with individual player preferences.

## 🛠️ Technical Architecture
The system utilizes a dual-model approach to ensure high-quality recommendations:
1. **Content-Based Filtering**: Uses **BGE (bge-base-en-v1.5)** for semantic embeddings and **Cosine Similarity**.
2. **Collaborative Filtering**: Implements **Singular Value Decomposition (SVD)** to discover latent user preferences.
3. **Hybrid Integration**: Combines scores to balance accuracy and item diversity.

## ✨ Key Features
* **Discover Mode**: Explore trending titles and "All-Time Legends".
* **Mix & Match Mode**: Filter by specific categories, mechanics, and complexity.
* **AI Picks**: Personalized recommendations based on your unique rating history.
* **Thai Language Support**: Specialized filtering for games available in Thai.
