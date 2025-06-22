from setuptools import setup, find_packages

setup(
    name='predictive_maintenance',
    version='0.1',
    packages=find_packages(where='src'),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "streamlit>=1.28.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "joblib>=1.2.0",
    ],
)
