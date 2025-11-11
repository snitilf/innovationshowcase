"""
test all required package imports for the project
"""

def test_import(package_name, import_statement):
    """test if a package can be imported"""
    try:
        exec(import_statement)
        print(f"✓ {package_name:<20} - SUCCESS")
        return True
    except ImportError as e:
        print(f"✗ {package_name:<20} - FAILED: {str(e)}")
        return False

print("Testing Package Imports...")
print("-" * 60)

# data collection and processing
test_import("wbdata", "import wbdata")
test_import("pandas", "import pandas as pd")
test_import("numpy", "import numpy as np")

# visualization
test_import("matplotlib", "import matplotlib.pyplot as plt")
test_import("seaborn", "import seaborn as sns")

# sentiment analysis
test_import("textblob", "from textblob import TextBlob")
test_import("vaderSentiment", "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer")

# machine learning
test_import("scikit-learn", "from sklearn.tree import DecisionTreeClassifier")
test_import("scikit-learn (model)", "from sklearn.model_selection import train_test_split")

# utilities
test_import("requests", "import requests")
test_import("jupyter", "import jupyter")

print("-" * 60)
