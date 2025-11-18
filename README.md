# The Global Trust Engine: The Creation of a ML4DI (Machine Learning for Development Integrity) Model Capable of Detecting and Combatting Fraud in the Development Arena

This project was developed as an optional assignment for the Student Innovation Showcase in **ECON 302: Money, Banking and Government Policy** at McGill University. The Student Innovation Showcase provides students with an opportunity to apply concepts from money, banking, and financial markets to real-world challenges, with the option to incorporate machine learning approaches.

## The Problem

Corruption in international development is a persistent challenge that undermines efforts to reduce poverty and promote economic growth. Approximately 30% of development funds are diverted before reaching recipient countries, creating a critical need for early warning systems that can identify high-risk environments before corruption scandals occur.

Traditional approaches to corruption detection rely heavily on governance indicators—measures of institutional quality that reflect conditions after weaknesses have already emerged. By the time governance metrics signal problems, corruption may have already caused significant damage. This project tests an alternative hypothesis: can economic indicators serve as early warning signals that identify vulnerable environments before governance metrics reflect institutional weaknesses?

## Project Overview

The Global Trust Engine is a machine learning model that identifies countries at high risk for corruption using economic indicators and sentiment analysis from news coverage. Rather than relying on governance metrics that only reflect problems after they've emerged, the model uses measurable economic conditions as alternative indicators that signal corruption risk.

The core hypothesis driving this work is that economic conditions can serve as leading indicators—signals that deteriorate before governance metrics show institutional weaknesses. Countries experiencing high poverty rates, excessive external debt burdens, or problematic government spending patterns may be more vulnerable to corruption, even if governance indicators haven't yet reflected these vulnerabilities.

The model achieves this by using a rigorous methodological approach: governance indicators determine corruption risk labels through a threshold-based classification system, but these same indicators are explicitly excluded from the predictive feature set. Instead, the model must learn to predict governance-based risk labels using only economic indicators and sentiment scores. This separation creates a genuine test of whether economic conditions can function as alternative indicators that signal institutional vulnerability.

## Methodology

The methodology centers on avoiding circular reasoning—a critical issue in machine learning where models simply memorize labeling rules rather than discovering meaningful patterns. Since governance indicators are used to create corruption risk labels (through a threshold-based system where countries with 4+ governance indicators below thresholds are classified as high-risk), using these same indicators as predictive features would cause the model to simply memorize the labeling rule.

To address this, governance indicators serve two distinct roles: they determine the target variable (corruption risk labels) and are retained for validation, but they are explicitly excluded from the predictive feature set. Five economic indicators and one sentiment score serve as predictive features, creating a rigorous test of whether these measures can predict governance-based labels.

A Decision Tree Classifier was selected for its interpretability, providing a transparent flowchart of decision-making rules that stakeholders can understand and trust. The algorithm uses gini impurity as the splitting criterion, which measures how mixed the classes are within a group. At each node, the algorithm selects the feature and threshold that minimizes gini impurity, creating the purest groups possible.

## Data Collection and Preparation

### Baseline Dataset Establishment

The project began by collecting standardized governance and economic indicators from the World Bank API for three case study countries: **Canada** (control country with consistently strong governance institutions), **Malaysia** (middle-income country with documented 1MDB corruption scandal, 2013-2015), and **Mozambique** (lower-income country with documented hidden debt crisis, 2013-2016). This selection creates a comparative framework spanning the spectrum from strong governance to moderate and weak governance with documented corruption.

Data collection focused on two categories of indicators:

**Governance Indicators**: Six Worldwide Governance Indicators (WGI) measuring institutional quality on a standardized scale (-2.5 to +2.5, higher scores indicate better governance):
- Voice and Accountability
- Political Stability and Absence of Violence
- Government Effectiveness
- Regulatory Quality
- Rule of Law
- Control of Corruption

**Economic Indicators**: Five economic indicators providing context on financial conditions:
- External Debt as Percentage of GNI
- Annual GDP Growth Rate
- Government Expenditure as Percentage of GDP
- Foreign Direct Investment Inflows as Percentage of GDP
- Poverty Headcount Ratio at $2.15 per Day

The 14-year timeframe (2010-2023) captures three phases relative to documented corruption cases: pre-scandal period (2010-2012), scandal period (2013-2016 for Mozambique, 2013-2015 for Malaysia), and post-scandal period (2017-2023).

### Risk Labeling Methodology

A systematic, threshold-based methodology classifies country-year observations as high-risk or low-risk for corruption. The methodology applies specific numerical thresholds to six World Bank governance indicators. Five indicators use threshold 1.15 (top 15-20% globally on the -2.5 to +2.5 scale), while Political Stability uses threshold 0.50 due to greater natural variation.

For each governance indicator, if a score falls below its threshold, a binary flag is set to 1 (weakness detected); if the score is at or above the threshold, the flag is set to 0 (strength maintained). These six binary flags are summed to create a total flag count, ranging from 0 (all indicators above thresholds) to 6 (all indicators below thresholds).

The final risk classification is determined by a simple threshold rule: if the total flag count is 4 or greater, the observation is classified as high-risk (corruption_risk = 1); otherwise, it is classified as low-risk (corruption_risk = 0). This methodology demonstrates perfect validation: it correctly classifies all documented corruption periods (Malaysia 2013-2015, Mozambique 2013-2016) as high-risk while maintaining Canada as low-risk throughout.

### Dataset Expansion

The baseline dataset was expanded from 3 countries to 19 countries, increasing observations from 42 to 266 country-year observations. The expansion adds 16 countries selected to span the full spectrum of governance quality:

- **High-Risk Countries**: Angola, Venezuela, Zimbabwe, Iraq, Ukraine
- **Medium-Risk Countries**: Brazil, South Africa, India, Philippines
- **Low-Risk Control Countries**: Norway, Denmark, Singapore, Australia, New Zealand, Switzerland, Germany

This selection ensures the model learns generalizable patterns across diverse contexts rather than memorizing country-specific patterns. The final labeled dataset contains 266 country-year observations with 112 (42.1%) classified as low-risk and 154 (57.9%) classified as high-risk, providing adequate representation for machine learning model training.

### Sentiment Analysis Validation

Sentiment analysis of corruption-related news articles was validated as a potential qualitative risk indicator. Sentiment analysis measures the emotional tone of written text, assigning numerical scores from -1 (extremely negative) to +1 (extremely positive). For corruption-related news, sentiment scores are expected to be negative because corruption events generate critical media coverage.

A two-source data collection strategy ensures comprehensive temporal coverage: Guardian API for historical coverage (2010-2016) and GDELT API for modern coverage (2017-2023). The combined dataset provides sentiment scores for 234 country-years (88% coverage), with missing values filled with neutral values (0.0) for analysis.

The validation revealed a counterintuitive but meaningful pattern: low-risk countries show more negative sentiment than high-risk countries. This reflects transparency mechanisms—in low-risk countries with free press, corruption incidents are more likely to be exposed and reported openly, generating more negative sentiment. In high-risk countries with media suppression, corruption may be hidden from public view, leading to less negative sentiment not because corruption is absent, but because it is hidden.

### Data Preparation for Model Training

The data preparation phase integrated governance indicators, economic indicators, and sentiment scores into a unified dataset for machine learning model training. Economic indicators showed 26.9% missing values overall, requiring a two-step imputation strategy: forward-filling within countries (using the most recent previous year's value) and median imputation for remaining gaps (using the median value across all countries). After imputation, all economic indicators showed complete coverage.

A stratified 80/20 train-test split divided the dataset: 80% for training (212 samples) and 20% for testing (54 samples). The stratified approach maintains the same proportion of high-risk and low-risk cases in both sets, preventing overfitting by evaluating whether the model learns generalizable patterns rather than country-specific memorization.

## Model Development

The Decision Tree Classifier was trained on 212 country-year observations with the following parameters: max_depth=5 (limits tree depth for interpretability), min_samples_split=10 and min_samples_leaf=5 (prevent overfitting by requiring minimum samples), class_weight='balanced' (handles class imbalance), and random_state=42 (ensures reproducibility).

The model automatically discovered that only three out of six predictive features are needed to achieve strong performance. The model identified three core economic indicators as the primary predictors of corruption risk:

1. **Poverty levels** (41.8% importance) - appearing at the root node of the decision tree
2. **External debt burden** (33.0% importance) - appearing at the second level
3. **Government spending patterns** (25.1% importance) - appearing at deeper levels

The model determined that GDP growth, foreign direct investment inflows, and sentiment scores do not improve predictions beyond what the three core indicators already provide. This feature reduction is a positive finding: the model identified that three core economic indicators are sufficient to predict corruption risk, making the model simpler and more interpretable while maintaining high accuracy.

The decision tree visualization provides a complete visual representation of how the model makes predictions, creating a transparent flowchart that enables stakeholders to understand and trust the model's decision-making process. Starting from the top (root node), the tree asks a series of yes/no questions about the economic indicators, with each answer leading to the next question until reaching a leaf node that provides the final prediction.

## Results and Validation

The trained model achieves strong performance on the test set, demonstrating its effectiveness as a risk classification system. Test set accuracy reaches 94.4%, with 93.6% recall and 96.7% precision. The small gap between training accuracy (93.9%) and test accuracy (94.4%) indicates the model learned generalizable patterns rather than memorizing training data.

The confusion matrix reveals that out of 54 test cases, only 3 were misclassified: 2 false negatives (high-risk countries missed) and 1 false positive (low-risk country incorrectly flagged). This low error rate confirms the model's reliability in identifying high-risk countries using economic warning signs.

The misclassified cases occurred when quantitative economic indicators didn't fully capture unique circumstances. The two false negatives involved Malaysia in 2015 and 2016, where low poverty rates may have masked other vulnerabilities. The single false positive involved Australia in 2011, where economic indicators suggested vulnerability but strong governance mechanisms prevented corruption.

These errors highlight the model's limitation: it cannot account for non-quantifiable factors such as historical legacies, cultural factors, or exceptional events that may create exceptions to general patterns. However, the low error rate demonstrates that economic indicators capture the majority of corruption risk patterns.

## Key Findings and Implications

The evaluation confirms the alternative indicator hypothesis: economic conditions serve as warning signs that identify vulnerable environments. The model successfully identifies high-risk countries using only economic indicators, demonstrating that measurable economic conditions can identify countries with elevated corruption risk.

The relationship between economic conditions and governance vulnerability aligns with theoretical frameworks that describe corruption as both a cause and consequence of economic vulnerability. Countries experiencing high poverty rates face increased pressure on limited resources, creating incentives for corruption. Excessive external debt burdens create fiscal stress that weakens institutional oversight. Problematic government spending patterns may reflect diversion of resources away from growth-promoting areas.

This enables development institutions to identify high-risk countries using economic indicators and allocate monitoring resources accordingly. Given that corruption diverts approximately 30% of development funds before reaching recipient countries, the model's ability to identify high-risk countries using economic warning signs is critical for development integrity.

The model's simplicity—requiring only three economic indicators—enhances practical utility, as policymakers can monitor these measures to assess corruption risk using annual country-level data. The decision tree's interpretability ensures that interventions are based on transparent, understandable criteria rather than opaque algorithmic decisions.

## Limitations and Future Work

The model cannot account for non-quantifiable factors such as historical legacies, cultural factors, or exceptional events that may create exceptions to general patterns. Additionally, sentiment analysis depends on news coverage availability (88% coverage rate), with missing values handled through neutral value imputation.

Future work could explore:
- Integration of additional data sources (e.g., financial transaction data, procurement records)
- Temporal modeling to capture dynamic risk patterns over time
- Country-specific model calibration for improved accuracy
- Integration with real-time monitoring systems for development institutions

## Project Structure

```
innovationshowcase/
├── data/
│   ├── raw/                    # Raw data from APIs
│   │   ├── corruption_data_baseline.csv
│   │   └── corruption_data_expanded.csv
│   ├── processed/              # Cleaned and labeled datasets
│   │   ├── corruption_data_labeled.csv
│   │   ├── corruption_data_expanded_labeled.csv
│   │   ├── final_training_data.csv
│   │   ├── train_set.csv
│   │   └── test_set.csv
│   └── sentiment/              # News sentiment scores
│       ├── news_headlines_raw.csv
│       └── sentiment_scores.csv
├── docs/                        # Documentation and reports
│   ├── InnovationShowcase.pdf
│   └── summaries/              # Notebook summaries
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
│   │   ├── decision_tree_diagram.png
│   │   ├── confusion_matrix_heatmap.png
│   │   ├── roc_curve.png
│   │   └── feature_importance_bar.png
│   └── tables/                 # Performance metrics
│       ├── model_performance.csv
│       └── feature_importance.csv
├── scripts/                    # Utility scripts
└── src/                        # Source code
    └── sentiment_analysis.py
```

## Technical Details

### Model Architecture

The model uses scikit-learn's `DecisionTreeClassifier` with the following parameters:
- **max_depth=5**: Limits tree depth for interpretability while maintaining predictive power
- **min_samples_split=10**: Requires minimum 10 samples to split a node, preventing overfitting
- **min_samples_leaf=5**: Requires minimum 5 samples in each leaf node, ensuring stable predictions
- **class_weight='balanced'**: Automatically adjusts weights to handle class imbalance (57.9% high-risk, 42.1% low-risk)
- **random_state=42**: Ensures reproducibility of results
- **criterion='gini'**: Uses gini impurity to measure class separation at each split

### Data Sources

- **World Bank API**: Governance indicators (6 WGI indicators) and economic indicators (5 indicators) for 19 countries, 2010-2023
- **Guardian API**: News articles for sentiment analysis, 2010-2016 (2,695 articles covering 102 country-year combinations)
- **GDELT API**: News articles for sentiment analysis, 2017-2023 (52,728 articles covering 132 country-year combinations)

### Technologies and Libraries

- **Python 3**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning model training and evaluation
- **matplotlib & seaborn**: Data visualization
- **wbdata**: World Bank API data retrieval
- **nltk**: Natural language processing for sentiment analysis

### Performance Metrics

- **Accuracy**: 94.4% (proportion of all predictions that are correct)
- **Precision**: 96.7% (when model predicts high-risk, it is correct 96.7% of the time)
- **Recall**: 93.6% (model correctly identifies 93.6% of all actual high-risk cases)
- **F1-Score**: 95.1% (harmonic mean of precision and recall)
- **ROC-AUC**: 95.9% (overall discriminative ability between high-risk and low-risk cases)

### Feature Importance

The model identified three core economic indicators with the following importance weights:
1. Poverty Headcount Ratio: 41.8%
2. External Debt as Percentage of GNI: 33.0%
3. Government Expenditure as Percentage of GDP: 25.1%

GDP Growth Rate, Foreign Direct Investment Inflows, and Sentiment Scores were not used by the model, suggesting these three core indicators are sufficient for accurate predictions.

## Conclusion

The Global Trust Engine represents a data-driven approach to development integrity, combining quantitative economic indicators with transparent machine learning to create a risk classification system. The model's strong performance (94.4% accuracy, 93.6% recall) and interpretability make it suitable for deployment in development contexts, where stakeholders require both accuracy and transparency to trust automated risk assessment systems.

By identifying high-risk environments where corruption vulnerabilities exist, the model enables policy interventions that can prevent the diversion of development funds and protect vulnerable populations from the consequences of institutional failure. The model addresses a critical gap in development integrity: traditional methods rely on governance indicators that reflect conditions after institutional weaknesses have already emerged. By using economic indicators that correlate with governance vulnerabilities, the model enables identification of high-risk environments using complementary indicators that provide an alternative assessment approach.

The technical contribution of this work lies in the rigorous separation of labeling features from predictive features, creating a genuine test of the alternative indicator hypothesis. The model's success demonstrates that economic conditions can predict governance-based risk labels, validating the core research question. The automatic feature selection that identified three core indicators demonstrates the model's ability to learn meaningful patterns while maintaining simplicity and interpretability.

These findings establish a foundation for data-driven risk classification systems in development contexts, where transparency and accuracy are both essential for building trust and enabling effective policy interventions.
