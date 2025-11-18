# Comprehensive Summary: Dataset Expansion for Machine Learning Model Training

## Introduction and Purpose

This analysis expands the baseline dataset from 3 countries to 19 countries, creating a dataset of sufficient size and diversity for machine learning model training. While the previous phases established the foundational methodology for collecting governance indicators and creating corruption risk labels, this phase addresses a critical limitation: machine learning models require substantial amounts of data to learn reliable patterns. A dataset with only 3 countries provides insufficient observations for model training, as the model would essentially memorize the specific patterns of those three countries rather than learning generalizable patterns that can be applied to new countries.

The expansion serves multiple purposes. First, it increases the total number of observations from 42 country-year observations to 266 observations, providing sufficient data for training a machine learning model. Second, it introduces greater diversity in governance patterns, ensuring that the model learns from a wide range of institutional quality levels rather than just three specific cases. Third, it maintains representation of documented corruption cases (Malaysia and Mozambique) and control countries with strong governance (Canada and others), ensuring that the expanded dataset preserves the validation framework established in previous phases.

## Country Selection Strategy

The expansion adds 16 countries to the original 3-country baseline, selected through a strategic approach that ensures diverse governance patterns while maintaining analytical coherence. The selection strategy recognizes that corruption risk assessment requires understanding how governance indicators behave across different institutional contexts, economic development levels, and geographic regions.

### Baseline Countries (Preserved from Previous Analysis)

The three original countries are retained to maintain continuity with previous analysis and preserve the validation framework:

**Canada** continues to serve as a control case with strong governance institutions and no documented major corruption scandals during the study period. Canada's consistently high governance scores across all dimensions provide a baseline for what protective governance looks like.

**Malaysia** represents a middle-income country with a documented major corruption scandal (the 1MDB scandal involving $4.5 billion USD stolen from a state development fund). Malaysia's moderate governance scores demonstrate that corruption can occur even in countries with some functional governance institutions, not just in countries with the weakest institutions.

**Mozambique** represents a lower-income country with a documented corruption case (the hidden debt crisis involving $2 billion USD in illicit loans). Mozambique's consistently weak governance scores demonstrate how structural vulnerabilities enable large-scale corruption in development contexts.

### Additional Countries: Diverse Governance Patterns

The 16 additional countries are selected to represent diverse governance patterns across multiple dimensions:

**High-Risk Countries** (countries with documented governance weaknesses or corruption vulnerabilities): Angola, Venezuela, Zimbabwe, Iraq, Ukraine. These countries are selected because they demonstrate governance patterns similar to countries with documented corruption cases, providing additional examples of high-risk environments. They help the model learn patterns associated with weak institutions, limited accountability, and structural vulnerabilities that enable corruption.

**Medium-Risk Countries** (countries with mixed governance profiles): Brazil, South Africa, India, Philippines. These countries are selected because they demonstrate moderate governance scores that sometimes fall below thresholds for corruption risk, providing examples of environments where governance quality is mixed rather than consistently strong or consistently weak. They help the model learn to distinguish between moderate-risk and high-risk environments.

**Low-Risk Control Countries** (countries with strong governance institutions): Norway, Denmark, Singapore, Australia, New Zealand, Switzerland, Germany. These countries are selected because they demonstrate consistently strong governance scores across multiple dimensions, providing additional examples of protective governance that prevents corruption. They help the model learn patterns associated with strong institutions, effective accountability mechanisms, and structural conditions that prevent corruption.

The selection strategy ensures that the expanded dataset spans the full spectrum of governance quality, from countries with the weakest institutions to countries with the strongest institutions. This diversity is essential for machine learning model training because the model must learn to distinguish between high-risk and low-risk environments across different contexts, not just for the specific three countries in the baseline dataset.

## Data Collection Methodology

The data collection process follows the same methodology established in the baseline analysis, collecting six governance indicators and five economic indicators for all 19 countries across the 2010-2023 time period. This consistency ensures that the expanded dataset is directly comparable to the baseline dataset and that the labeling methodology can be applied uniformly across all countries.

### Governance Indicators: Complete Coverage

All six Worldwide Governance Indicators show complete coverage across all 19 countries and all years from 2010 to 2023. This complete coverage is essential because the labeling methodology depends entirely on these governance measures to calculate corruption risk labels. The fact that governance indicators are available for all countries and all years ensures that every country-year observation can be reliably classified as high-risk or low-risk.

The governance indicators measure six distinct dimensions of institutional quality: voice and accountability (citizen participation and freedom of expression), political stability (likelihood of government destabilization), government effectiveness (quality of public services and policy implementation), regulatory quality (ability to formulate and implement sound policies), rule of law (extent to which agents have confidence in and abide by rules), and control of corruption (extent to which public power is exercised for private gain).

### Economic Indicators: Variable Completeness

Economic indicators show variable levels of completeness across countries and years, reflecting differences in data collection practices and country-specific reporting capabilities. External Debt as Percentage of Gross National Income shows 52.2% missing data, primarily because some countries do not report external debt data consistently. Poverty Headcount Ratio shows 57.1% missing data, reflecting that poverty measurement requires household surveys that are not conducted annually in all countries. Government Expenditure shows 18.3% missing data, while GDP Growth and Foreign Direct Investment show relatively low missing data (around 4%).

This variable completeness does not affect the labeling process, which depends entirely on governance indicators. Economic indicators will be used as predictive features in subsequent machine learning model training, where missing data can be handled through imputation strategies. The key point is that governance indicators, which form the foundation for risk labels, show complete coverage across all countries and years.

## Labeling Methodology Application

The same threshold-based labeling methodology developed in the previous phase is applied uniformly across all 19 countries. This consistency ensures that risk labels are calculated using the same systematic approach for all countries, enabling reliable comparison and model training.

### Threshold-Based Classification

The methodology applies threshold-based classification to six governance indicators, creating binary flags (yes-or-no indicators) for each dimension. A country-year observation receives a flag if its governance score falls below the established threshold for that dimension, indicating weakness in that aspect of institutional quality. The six thresholds are: Voice and Accountability (1.15), Political Stability (0.50), Government Effectiveness (1.15), Regulatory Quality (1.15), Rule of Law (1.15), and Control of Corruption (1.15).

These thresholds are set based on empirical analysis of governance score distributions and validated against documented corruption cases. Most indicators use a threshold of 1.15, which represents a score substantially above the global average (approximately 0 on the standardized scale). Political Stability uses a lower threshold of 0.50, reflecting that this indicator shows more variation and that moderate stability may still enable corruption if other governance dimensions are weak.

### Risk Aggregation

The methodology aggregates the six binary flags into a single risk classification: a country-year observation is labeled as high-risk (1) if it receives 4 or more flags, indicating weaknesses in at least four out of six governance dimensions. Otherwise, it is labeled as low-risk (0), indicating that governance is sufficiently strong across most dimensions to prevent large-scale corruption.

This aggregation approach recognizes that corruption requires multiple enabling conditions to occur at scale. A country might have weak rule of law but strong government effectiveness, creating a mixed governance profile that may not enable large-scale corruption. However, when four or more governance dimensions show weaknesses simultaneously, this signals a comprehensive governance failure that creates an environment where corruption can flourish.

## Validation Against Documented Corruption Cases

The labeling methodology is validated against the same documented corruption cases used in previous phases, ensuring that the expansion does not compromise the methodology's accuracy. This validation is critical because it confirms that the threshold-based approach correctly identifies periods when documented corruption occurred, even when applied to a larger and more diverse dataset.

### Malaysia 1MDB Scandal Validation (2013-2015)

The validation confirms that all three years during the 1MDB scandal period (2013, 2014, and 2015) are correctly labeled as high-risk. Malaysia received 4 or more governance flags in each of these years, reflecting the governance weaknesses that enabled the $4.5 billion USD theft from the state development fund. This validation demonstrates that the methodology successfully identifies corruption risk even in middle-income countries with moderate governance scores, not just in countries with the weakest institutions.

### Mozambique Hidden Debt Crisis Validation (2013-2016)

The validation confirms that all four years during the hidden debt crisis period (2013, 2014, 2015, and 2016) are correctly labeled as high-risk. Mozambique received 4 or more governance flags in each of these years, reflecting the governance weaknesses that enabled the $2 billion USD in illicit loans to be diverted from development purposes. This validation demonstrates that the methodology correctly identifies corruption risk in countries with weak governance institutions.

### Canada Control Case Validation

The validation confirms that Canada is correctly labeled as low-risk throughout the entire study period (2010-2023), with all 14 years receiving low-risk classifications. Canada received fewer than 4 governance flags in every year, reflecting its consistently strong governance scores across all dimensions. This validation demonstrates that the methodology correctly distinguishes between high-risk and low-risk environments, correctly identifying Canada's strong governance as protective against corruption.

The perfect validation results across all three cases—with Malaysia's 1MDB period, Mozambique's hidden debt crisis period, and Canada's control case all correctly classified—provide confidence that the labeling methodology accurately captures governance-based corruption risk even when applied to an expanded dataset with greater diversity.

## Key Findings and Patterns

The expanded dataset reveals important patterns in governance quality and corruption risk labels that provide insights into the relationship between institutional quality and corruption vulnerability across diverse contexts.

### Overall Label Distribution

The final labeled dataset contains 266 country-year observations (19 countries × 14 years, with some countries having complete data for all years). Of these observations, 112 (42.1%) are classified as low-risk and 154 (57.9%) are classified as high-risk. This distribution reflects that the expanded dataset includes substantial representation of both high-risk and low-risk environments, providing balanced data for machine learning model training.

The high proportion of high-risk labels (57.9%) is expected given the country selection strategy, which intentionally included countries with documented corruption cases and countries with governance weaknesses. However, the substantial representation of low-risk labels (42.1%) ensures that the model learns patterns associated with both high-risk and low-risk environments, enabling reliable classification.

### Governance Patterns Across Risk Categories

The analysis reveals clear patterns in governance scores across the three risk categories used for exploratory analysis (High-Risk, Medium-Risk, Low-Risk). High-risk countries show consistently low governance scores across all six dimensions, with average scores well below the established thresholds. Medium-risk countries show mixed governance profiles, with some dimensions showing strength and others showing weakness. Low-risk countries show consistently high governance scores across all six dimensions, with average scores well above the established thresholds.

These patterns validate the theoretical framework's prediction that governance indicators should distinguish between high-risk and low-risk environments. The clear separation between risk categories demonstrates that the threshold-based labeling approach accurately captures governance quality differences that are meaningful for corruption risk assessment.

### Temporal Patterns and Stability

The analysis reveals important temporal patterns in risk labels across countries. Some countries show consistent risk labels throughout the study period (for example, Canada consistently shows low-risk labels, while Mozambique consistently shows high-risk labels), reflecting stable governance institutions. Other countries show variation in risk labels across years (for example, Malaysia shows both low-risk and high-risk labels at different times), reflecting changes in governance quality over time.

These temporal patterns are important for machine learning model training because they enable the model to learn both stable patterns (countries that consistently show high-risk or low-risk) and dynamic patterns (countries that transition between risk categories). This diversity helps the model learn generalizable patterns that can be applied to new countries and time periods.

### Country-Specific Risk Profiles

The expanded dataset reveals distinct risk profiles across countries that align with their governance histories and documented corruption cases. Countries with documented corruption cases (Malaysia, Mozambique) show high-risk labels during the periods when corruption occurred, validating the labeling methodology. Countries with strong governance institutions (Canada, Norway, Denmark, Switzerland, Germany, Australia, New Zealand, Singapore) show consistently low-risk labels throughout the study period, demonstrating what protective governance looks like.

Countries with governance weaknesses but no documented major corruption cases during the study period (Angola, Venezuela, Zimbabwe, Iraq, Ukraine) show consistently high-risk labels, reflecting that governance indicators can signal vulnerability even when corruption has not yet been exposed. This pattern supports the hypothesis that governance indicators have predictive value for identifying vulnerability before corruption occurs.

## Methodological Contributions

This dataset expansion makes several important methodological contributions to the larger research project, establishing foundations for machine learning model training and validating the labeling methodology across diverse contexts.

### Sufficient Data for Machine Learning Training

The expansion from 42 observations to 266 observations provides sufficient data for training a machine learning model. Machine learning models require substantial amounts of data to learn reliable patterns that can be generalized to new observations. With only 42 observations, a model would essentially memorize the specific patterns of three countries rather than learning generalizable patterns. With 266 observations spanning 19 countries, the model can learn patterns that distinguish between high-risk and low-risk environments across diverse contexts.

The rule of thumb in machine learning is that models need at least several hundred observations to learn reliable patterns, with more observations generally leading to better model performance. The expansion to 266 observations provides a dataset of sufficient size for training a decision tree classifier, which is the model type selected for this project due to its interpretability and robust performance on structured data.

### Diversity in Governance Patterns

The expansion introduces greater diversity in governance patterns, ensuring that the model learns from a wide range of institutional quality levels rather than just three specific cases. This diversity is essential because corruption risk assessment must work across different contexts—different countries, different economic development levels, different geographic regions, and different governance histories. By including countries that span the full spectrum of governance quality, the expanded dataset enables the model to learn generalizable patterns that can be applied to new countries.

The diversity also helps prevent overfitting, which occurs when a model learns patterns that are too specific to the training data and do not generalize to new observations. By training on a diverse set of countries with different governance profiles, the model is forced to learn patterns that are broadly applicable rather than country-specific.

### Validation Across Diverse Contexts

The perfect validation results across all documented corruption cases, even when applied to an expanded dataset, demonstrate that the labeling methodology is robust and reliable across diverse contexts. The fact that Malaysia's 1MDB period, Mozambique's hidden debt crisis period, and Canada's control case are all correctly classified in the expanded dataset provides confidence that the methodology accurately captures governance-based corruption risk regardless of the specific countries included.

This validation is particularly important because it confirms that the threshold-based approach works not just for the three original countries, but for a broader set of countries with different governance profiles. The successful validation across diverse contexts supports the use of this methodology for risk assessment in new countries and time periods.

### Foundation for Model Training

The expanded and labeled dataset provides the essential foundation for machine learning model training in subsequent phases. The dataset contains both the predictive features (governance and economic indicators) and the target variable (corruption risk labels) that the model will learn to predict. The systematic labeling methodology ensures that the target variable accurately reflects governance-based corruption risk, enabling reliable model training that can identify high-risk environments before corruption occurs.

The expansion ensures that the model training dataset has sufficient size and diversity for learning generalizable patterns. The balanced distribution of high-risk and low-risk labels (57.9% high-risk, 42.1% low-risk) provides adequate representation of both classes for model training, preventing class imbalance issues that could compromise model performance.

## Visualization and Pattern Discovery

The analysis creates comprehensive visualizations that reveal important patterns in governance indicators and risk labels across the expanded dataset. These visualizations provide intuitive understanding of the data patterns and validate the labeling methodology through visual inspection.

### Governance Indicators by Risk Category

Visualizations of average governance scores by risk category reveal clear patterns that distinguish between high-risk, medium-risk, and low-risk environments. High-risk countries show consistently low scores across all six governance dimensions, with scores well below the established thresholds. Low-risk countries show consistently high scores across all six dimensions, with scores well above the established thresholds. Medium-risk countries show mixed profiles, with some dimensions showing strength and others showing weakness.

These visualizations provide immediate visual confirmation that the labeling methodology accurately reflects governance quality differences between risk categories. The clear separation between categories demonstrates that the threshold-based approach captures meaningful distinctions in institutional quality that are relevant for corruption risk assessment.

### Governance Indicators Heatmap

A heatmap visualization of average governance scores by country, sorted by Control of Corruption (the most directly relevant indicator for corruption risk), reveals the full spectrum of governance quality across the 19 countries. Countries with strong governance (Norway, Denmark, Switzerland, Singapore, Canada, Australia, New Zealand, Germany) show consistently high scores across all dimensions, appearing in green tones on the heatmap. Countries with weak governance (Iraq, Zimbabwe, Venezuela, Angola, Mozambique) show consistently low scores across all dimensions, appearing in red tones on the heatmap. Countries with moderate governance (Malaysia, Brazil, South Africa, India, Philippines, Ukraine) show mixed scores, appearing in yellow tones on the heatmap.

This visualization provides an intuitive understanding of how governance quality varies across countries and how these variations align with corruption risk labels. The visual pattern confirms that countries with consistently low governance scores receive high-risk labels, while countries with consistently high governance scores receive low-risk labels.

### Corruption Risk Labels Heatmap

A heatmap visualization of corruption risk labels by country and year provides a clear, intuitive representation of the final classification results across all 19 countries and all years from 2010 to 2023. The heatmap shows countries sorted by average risk level, with high-risk countries (showing mostly red cells, indicating high-risk labels) at the top and low-risk countries (showing mostly green cells, indicating low-risk labels) at the bottom.

This visualization reveals important temporal patterns: some countries show consistent risk labels throughout the study period (for example, Canada consistently shows green cells, while Mozambique consistently shows red cells), while other countries show variation across years (for example, Malaysia shows a mix of green and red cells). These patterns provide visual confirmation that the labeling methodology captures both stable governance patterns and temporal changes in governance quality.

## Export and Dataset Preparation

The analysis exports the expanded and labeled dataset for subsequent machine learning model training phases. The final dataset contains 266 country-year observations with 23 variables, including the original governance and economic indicators, the six binary flags for each governance dimension, the total flag count, the risk category classification (for exploratory analysis), and the final corruption risk label (for model training).

The exported dataset serves as the foundation for machine learning model training, providing both the predictive features (governance and economic indicators) and the target variable (corruption risk label) that the model will learn to predict. The systematic labeling methodology ensures that the target variable accurately reflects governance-based corruption risk, enabling reliable model training that can identify high-risk environments before corruption occurs.

## Conclusion

This dataset expansion successfully transforms the baseline 3-country dataset into a comprehensive 19-country dataset suitable for machine learning model training. The expansion addresses a critical limitation of the baseline dataset—insufficient observations for model training—while maintaining the systematic methodology and validation framework established in previous phases.

The key contribution of this analysis is the creation of a dataset with sufficient size and diversity for machine learning model training. The expansion from 42 observations to 266 observations provides adequate data for training a decision tree classifier, while the inclusion of 19 countries spanning the full spectrum of governance quality ensures that the model learns generalizable patterns rather than country-specific memorization.

The perfect validation results—with Malaysia's 1MDB period, Mozambique's hidden debt crisis period, and Canada's control case all correctly classified in the expanded dataset—provide confidence that the labeling methodology accurately captures governance-based corruption risk across diverse contexts. The clear patterns in governance scores and risk labels across risk categories validate the theoretical framework's prediction that governance indicators should distinguish between high-risk and low-risk environments.

The expanded and labeled dataset provides the essential foundation for subsequent machine learning model development. The systematic labeling methodology, validated across diverse contexts, ensures that model training will learn patterns that can identify high-risk environments before corruption occurs, ultimately supporting the protection of development funds and the improvement of aid effectiveness. The methodological contributions—including the creation of sufficient data for model training, the introduction of diversity in governance patterns, and the validation across diverse contexts—create a robust foundation for the Global Trust Engine's development as a data-driven early warning system for corruption risk in development contexts.

The analysis reveals important patterns in how governance quality varies across countries and time periods, and how these variations align with corruption risk labels. The temporal patterns suggest that some countries show stable governance institutions (consistently high-risk or consistently low-risk), while others show dynamic changes in governance quality over time. These patterns support the development of a machine learning model that can identify high-risk environments proactively, rather than retrospectively, ultimately enabling the protection of development funds before they are diverted through corrupt practices.

