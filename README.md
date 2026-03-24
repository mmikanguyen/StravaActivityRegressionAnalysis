# Strava Runs Regression Model 🏃‍♂️

Predict running performance based on Strava activity data using machine learning.

# Project Overview

This project uses running data from the Strava API
 to build a regression model that predicts total running time based on metrics such as distance, heart rate, cadence, elevation, and recent training load. The goal is to learn patterns from past runs and estimate future performance across different distances.

# Context

This project is based on my personal running journey. Over the past six months, I have transitioned from a beginner to a more structured runner. My training evolved from short, easy runs to longer runs, tempo workouts, and progressively higher mileage. This project captures that evolution and uses it to better predict performance trends.

# How It Works

## 1. Fetch Data from Strava
Use your Strava refresh token to obtain an access token and pull your recent running activities via the Strava API.
Activities are saved locally in runs.csv

## 2. Features
features.csv contains additional variables calculated for modeling: 
- distance_km: total distance in kilometers
- total_time_min: total moving time in minutes
- pace_per_km: average pace in min/km
- avg_hr: average heart rate during the run
- avg_cadence: average running cadence
- elevation_per_km: elevation gain per kilometer
- weekly_km: total distance run in past 7 days
- rolling_pace: average pace over recent runs
- hr_percent_max: average heart rate as a % of est. max heart rate
- effort_pace: pace adjusted to heart rate

## 3. Model Training
Trains and compares multiple regression models including:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
Evaluates performance using MAE, RMSE, and R².
Uses an 80/20 train-test split, with the most recent runs reserved for testing.

## 4. Exploration & Visualization
Run explore.ipynb to train all models, generate visualizations (Actual vs Predicted, Residuals, Feature Distributions) and compare model performance side-by-side and identify the best performing model

## Setup
Clone repo and create a .env file in the project root with:
- STRAVA_CLIENT_ID=your_client_id
- STRAVA_CLIENT_SECRET=your_client_secret
- STRAVA_REFRESH_TOKEN=your_refresh_token

Run model in explore.ipynb to test model generate visualizations.
