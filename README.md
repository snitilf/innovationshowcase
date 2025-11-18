# Innovation Showcase - ML4DI Model

## Overview

This project builds a machine learning model to detect corruption risk in development aid contexts. The goal is to create an early warning system that can identify countries at high risk for corruption before scandals occur, helping protect development funds and improve aid effectiveness.

The model uses economic indicators and sentiment analysis from news coverage to predict corruption risk, rather than relying on governance metrics that only reflect problems after they've already emerged. This approach tests whether economic conditions can serve as "leading indicators" that deteriorate before governance metrics show institutional weaknesses.

## What We're Building

The theoretical framework explains that corruption thrives in environments with limited accountability, weak enforcement systems, and institutional weaknesses. Traditional methods struggle with the complexity and scale of corruption detection, which is why machine learning approaches are needed.

Our "Global Trust Engine" model aims to:
- Identify high-risk environments before corruption scandals occur
- Use measurable economic indicators as early warning signals
- Provide transparent, interpretable predictions that stakeholders can trust
- Support policy decisions for development integrity

The model is trained on data from 19 countries spanning 2010-2023, including documented corruption cases (Malaysia's 1MDB scandal, Mozambique's hidden debt crisis) and control countries with strong governance (Canada, Norway, Denmark, etc.).

## What's Been Done So Far

### 1. Data Collection (Notebook 01)

Started by collecting governance and economic indicators from the World Bank API for three baseline countries: Canada (control), Malaysia (1MDB scandal), and Mozambique (hidden debt crisis). Collected six Worldwide Governance Indicators (voice and accountability, political stability, government effectiveness, regulatory quality, rule of law, control of corruption) plus five economic indicators (GDP growth, external debt, government expenditure, FDI inflows, poverty rates) for 2010-2023.

### 2. Data Cleaning and Labeling (Notebook 02)

Created corruption risk labels using a threshold-based approach. Each of the six governance indicators gets a flag if it falls below a threshold (e.g., control of corruption < 1.15). Countries with 4 or more flags are labeled as high-risk (1), others as low-risk (0). This methodology correctly labels the documented corruption periods: Malaysia 2013-2015 (1MDB) and Mozambique 2013-2016 (hidden debt) as high-risk, while Canada remains low-risk throughout.

### 3. Dataset Expansion (Notebook 03)

Expanded from 3 countries to 19 countries to get enough data for machine learning. Added countries representing diverse governance patterns: high-risk countries (Angola, Venezuela, Zimbabwe, Iraq, Ukraine), medium-risk countries (Brazil, South Africa, India, Philippines), and additional low-risk controls (Norway, Denmark, Singapore, Australia, New Zealand, Switzerland, Germany). Final dataset has 266 country-year observations across 2010-2023.

### 4. Sentiment Analysis Validation (Notebook 04)

Collected and analyzed sentiment scores from corruption-related news articles. Used two data sources: Guardian API (2010-2016) and GDELT API (2017-2023) to ensure comprehensive coverage. Found that both high-risk and low-risk countries show negative sentiment (corruption news is inherently negative), but the pattern reveals transparency: countries with free press show more negative sentiment (corruption gets exposed), while countries with media suppression show less negative sentiment (corruption is hidden). This validates sentiment as a qualitative early warning indicator.

### 5. Data Preparation (Notebook 05)

Integrated all data sources and prepared the dataset for model training. Used a key methodological approach: governance indicators determine the risk labels, but are excluded from the predictive features to avoid circular reasoning. Instead, the model uses five economic indicators and one sentiment score as features, testing whether these can predict governance-based labels. This tests the "leading indicator" hypothesis - that economic conditions deteriorate before governance metrics reflect problems.

Handled missing data using forward-fill within countries and median imputation. Split the data into training (212 samples) and test (54 samples) sets with maintained class balance.

### 6. Model Training (Notebook 06)

Trained a Decision Tree Classifier chosen for its interpretability. The model automatically identified that only three features are needed: poverty levels (41.8% importance), external debt burden (33.0% importance), and government spending patterns (25.1% importance). GDP growth, FDI inflows, and sentiment scores weren't used by the model, suggesting these three core economic indicators are sufficient.

The decision tree provides a clear flowchart showing how different economic thresholds lead to risk classifications, making it transparent and trustworthy for policy decisions.

### 7. Model Evaluation (Notebook 07)

Evaluated the trained model on the test set. Achieved strong performance:
- 94.4% accuracy
- 96.7% precision (when it flags high-risk, it's usually correct)
- 93.6% recall (catches most actual high-risk cases - critical for early warning)
- 95.1% F1-score
- 95.9% ROC-AUC (excellent ability to distinguish high-risk from low-risk)

The model correctly identifies the documented corruption cases and shows good generalization (small gap between training and test performance). Error analysis shows only 2 false negatives (Malaysia 2015-2016, which had unusually low poverty rates despite high governance risk) and 1 false positive (Australia 2011, which had similar economic indicators to high-risk countries but strong governance prevented corruption).

## Current Status

The model is trained and validated, demonstrating that economic indicators can reliably serve as leading indicators for corruption risk. The three core indicators (poverty, external debt, government spending) capture economic conditions that signal vulnerability before governance metrics reflect institutional weaknesses.

Key findings:
- Economic indicators can predict governance-based labels, supporting the leading indicator hypothesis
- Three core indicators are sufficient for accurate predictions
- The model achieves high recall, critical for an early warning system
- Decision tree structure provides transparent, interpretable predictions

## Project Structure

```
innovationshowcase/
├── data/
│   ├── raw/                    # Raw data from APIs
│   ├── processed/              # Cleaned and labeled datasets
│   └── sentiment/              # News sentiment scores
├── docs/
│   └── theory.txt              # Theoretical framework
├── models/
│   ├── decision_tree_model.pkl # Trained model
│   └── feature_names.txt       # Feature definitions
├── notebooks/
│   ├── 01_worldbank_api_exploration.ipynb
│   ├── 02_data_cleaning_labeling.ipynb
│   ├── 03_expand_dataset_countries.ipynb
│   ├── 04_validate_sentiment_sources.ipynb
│   ├── 05_data_preparation.ipynb
│   ├── 06_decision_tree_training.ipynb
│   └── 07_model_evaluation.ipynb
├── results/
│   ├── figures/                # Visualizations
│   └── tables/                 # Performance metrics
└── scripts/                    # Utility scripts
```

## Technical Notes

The model uses scikit-learn's DecisionTreeClassifier with:
- max_depth=5 (for interpretability)
- min_samples_split=10, min_samples_leaf=5 (to prevent overfitting)
- class_weight='balanced' (to handle class imbalance)
- random_state=42 (for reproducibility)

All analysis is done in Python using pandas, scikit-learn, matplotlib, and seaborn. Data is collected from World Bank API (governance and economic indicators), Guardian API (news 2010-2016), and GDELT API (news 2017-2023).
