## Strava Runs Regression Model 🏃‍♂️

Predict running performance based on Strava activity data using machine learning.

Project Overview

This project uses running data from the Strava API
 to build a regression model that predicts total running time based on metrics such as distance, heart rate, cadence, elevation, and recent training load. The goal is to learn patterns from past runs and estimate future performance across different distances.

## Context

This project is based on my personal running journey. Over the past six months, I have transitioned from a beginner to a more structured runner. My training evolved from short, easy runs to longer runs, tempo workouts, and progressively higher mileage. This project captures that evolution and uses it to better predict performance trends.

## How It Works
Fetch Data From Strava
Use your Strava refresh token to obtain an access token to pull your recent activities from the Strava API. This project saves running activities in runs.csv.

## Features
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

## Model
Trains a Linear Model to predict total_time_min and evaluates performance through MAE, MSE, and R2. Uses 80/20 split and uses most recent activities for testing.

## Setup
Clone repo and create a .env file in the project root with:
- STRAVA_CLIENT_ID=your_client_id
- STRAVA_CLIENT_SECRET=your_client_secret
- STRAVA_REFRESH_TOKEN=your_refresh_token

Run model in explore.ipynb to test model generate visualizations.
