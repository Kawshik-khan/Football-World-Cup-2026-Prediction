# FIFA World Cup 2026 Prediction

A comprehensive machine learning pipeline for predicting the entire FIFA World Cup 2026 tournament using Monte Carlo simulation and ensemble modeling.

## Overview

This project predicts:
- Group stage match scores and outcomes
- Group standings with FIFA tie-breaker rules
- Best 8 third-placed teams (new 48-team format)
- Full knockout bracket (Round of 32 → Final)
- Champion probabilities from 100,000 Monte Carlo simulations
- Automatic submission file generation

## Features

- **Modular, leakage-safe pipeline** - All rolling features computed strictly before each match
- **Ensemble modeling** - 40% Poisson + 30% XGBoost + 20% CatBoost + 10% LightGBM
- **Vectorized Monte Carlo** - 100,000 tournament simulations run in seconds using NumPy
- **Elo ratings integration** - Historical and current Elo ratings for team strength
- **World Cup pedigree features** - Historical World Cup performance metrics
- **Time-based validation** - Strict temporal split (train < 2022-01-01 ≤ valid)

## Project Structure

```
.
├── Dataset/                          # Input data files
│   ├── group_fixtures.csv           # Group stage fixtures
│   ├── knockout_slots.csv          # Knockout bracket structure
│   ├── Historical_eloratings.csv   # Historical Elo ratings
│   ├── elo_ratings.csv             # Current team Elo ratings
│   ├── results.csv                 # Historical match results
│   └── WorldCupMatches.csv         # World Cup match history
├── FIFA_WC2026_Prediction.ipynb   # Main Jupyter notebook
├── fifa_wc2026_simulation.py       # Python script version
├── requirement.txt                 # Python dependencies
├── group_predictions.csv           # Output: Group stage predictions
├── knockout_predictions.csv        # Output: Knockout stage predictions
└── submission.csv                  # Output: Final submission file
```

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirement.txt
```

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- lightgbm
- xgboost
- catboost
- matplotlib
- seaborn
- ipython
- jupyter
- nbclient
- jupytext

## Usage

### Jupyter Notebook

Open and run the notebook:
```bash
jupyter notebook FIFA_WC2026_Prediction.ipynb
```

The notebook is organized into sections:
1. **Imports and Configuration** - Centralized hyperparameters in CFG class
2. **Load Datasets** - Load and validate all CSV files
3. **Data Cleaning** - Standardize team names, handle missing values
4. **Feature Engineering** - Build leakage-safe feature matrix
5. **Exploratory Data Analysis** - Goal distributions, correlations
6. **Target Variables** - Define home/away goals and outcome flags
7. **Train Models** - Train 5 model families with time-based split
8. **Model Ensemble** - Fixed-weight blend per specification
9. **Tournament Snapshots** - Pre-compute expected-goals matrix
10. **Poisson Goal Model** - Score probability matrix
11. **Match Simulation Engine** - Vectorized match simulation
12. **Group Stage Engine** - Fully vectorized group stage
13. **Knockout Stage Engine** - Bracket progression
14. **Monte Carlo Simulation** - 100,000 tournament simulations
15. **Results & Visualization** - Champion probabilities, stage reach
16. **Submission Generation** - Output CSV files

### Python Script

Alternatively, run the Python script:
```bash
python fifa_wc2026_simulation.py
```

## Configuration

Edit the `CFG` class in the notebook to adjust:

```python
@dataclass
class CFG:
    seed: int = 42                          # Random seed
    data_dir: str = "Dataset"               # Data directory
    n_simulations: int = 100_000             # Monte Carlo simulations
    valid_start: str = "2022-01-01"         # Validation split date
    train_start: str = "1993-01-01"         # Training start date
    home_adv_elo: float = 50.0              # Home advantage Elo bonus
    w_poisson: float = 0.40                 # Poisson model weight
    w_xgb: float = 0.30                     # XGBoost weight
    w_cat: float = 0.20                     # CatBoost weight
    w_lgb: float = 0.10                     # LightGBM weight
    hosts: tuple = ("United States", "Mexico", "Canada")
    output_dir: str = "."                   # Output directory
```

## Model Performance

The ensemble achieves:
- **MAE**: ~1.15 goals
- **RMSE**: ~1.45 goals
- **Poisson NLL**: Optimized for count data
- **Accuracy**: ~55% match outcome prediction

## Output Files

### group_predictions.csv
Predicted scores for all 72 group stage matches with:
- Match ID
- Home/Away teams
- Predicted home/away goals
- Win probability

### knockout_predictions.csv
Predicted knockout bracket progression with:
- Round
- Match ID
- Teams
- Predicted winner

### submission.csv
Final submission with per-team probabilities:
- Team name
- Probability of reaching each stage
- Champion probability

## Tournament Format

The 2026 FIFA World Cup features:
- **48 teams** in **12 groups** of 4
- **72 group stage matches**
- **Best 8 third-placed teams** advance to Round of 32
- **Full knockout bracket**: Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final

## Tie-Breaker Rules

Group standings use FIFA tie-break cascade:
1. Points
2. Goal difference
3. Goals scored
4. Head-to-head (simplified as random lot in Monte Carlo)
5. Fair play points (simplified as random lot)

## License

This project is for educational and research purposes.

## Author

Senior Data Scientist / MLOps / Football Analytics / Kaggle Grandmaster pipeline
