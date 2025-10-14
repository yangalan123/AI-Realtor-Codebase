# AI Realtor: Towards Grounded Persuasive Language Generation for Automated Copywriting

This repository contains the codebase for the paper [**"AI Realtor: Towards Grounded Persuasive Language Generation for Automated Copywriting"**](https://arxiv.org/abs/2502.16810).

## Citation

If you use this code as part of any published research, please acknowledge the following paper:

```bibtex
@article{wu2025grounded,
  title={AI Realtor: Towards Grounded Persuasive Language Generation for Automated Copywriting},
  author={Wu, Jibang and Yang, Chenghao and Wu, Yi and Mahns, Simon and Wang, Chaoqi and Zhu, Hao and Fang, Fei and Xu, Haifeng},
  journal={arXiv preprint arXiv:2502.16810},
  year={2025}
}
```

## Data Release

The datasets used in this research are available at:

- **User Preference Data**: https://huggingface.co/datasets/Sigma-Lab/AI_Realtor_User_Preference_Anonymized
- **Listing Data**: https://huggingface.co/datasets/Sigma-Lab/AI_Realtor_Listing_Data

**Important**: Users must agree to the license terms before accessing the datasets.

## License and Usage

This project is intended **only for educational and research purposes, not for commercial purposes**.

## Privacy Disclaimer

Reasonable efforts have been made to process the data and remove or anonymize Personally Identifiable Information (PII). However, the complete absence of PII cannot be guaranteed. The User agrees to handle the Dataset with care and is solely responsible for:

- Ensuring their use of the Dataset complies with all applicable privacy laws and regulations (e.g., GDPR, CCPA).
- Any consequences arising from the use of any PII that may remain within the Dataset.
- Not attempting to re-identify any individuals from the anonymized data.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

You need to set up your OpenAI API key in the relevant files. Replace `"YOUR_KEY"` with your actual OpenAI API key in the following files:

- `user_simulation/predicting_preference_batch_api.py` (line 4)
- `user_simulation/predict_preference.py` (line 8)
- `hallucination_detection/check_match_gpt4_soft_match.py` (line 3)
- `hallucination_detection/check_exact_match_rule_em.py` (line 3)
- `rag_agents/preference_summary_from_ranking_demo.py` (line 7)

For example:
```python
os.environ["OPENAI_API_KEY"] = "your-actual-api-key-here"
```

### 3. Initialize Data

Run the following command to create the initial data:

```python
from utils import get_original_all_features_data
all_features = get_original_all_features_data()
```

This will download the listing data from Hugging Face and save it to `./data/ai_realtor_listing_data.json`.

## Project Structure

```
├── benchmark/                    # Evaluation and benchmarking scripts
├── hallucination_detection/      # Hallucination detection and evaluation
├── highlight_model/             # Highlight model training and inference
├── rag_agents/                  # RAG-based agent implementations
├── user_simulation/             # User preference simulation and prediction
├── const.py                     # Constants and feature mappings
├── utils.py                     # Utility functions
└── requirements.txt             # Python dependencies
```

## Key Components

### Feature Processing
- `const.py`: Contains the desired feature names and mappings from original features to standardized ones
- `utils.py`: Utility functions for data processing, feature normalization, and data loading

### User Simulation
- `user_simulation/`: Contains scripts for predicting user preferences and simulating user behavior

### Highlight Model
- `highlight_model/`: Training and inference scripts for the highlight model that identifies important features

### RAG Agents
- `rag_agents/`: Retrieval-Augmented Generation agents for generating persuasive real estate descriptions

### Evaluation
- `benchmark/`: Scripts for evaluating model performance using ELO ratings and win rates
- `hallucination_detection/`: Tools for detecting and evaluating hallucination in generated content

## Usage Examples

### Loading Data
```python
from utils import get_original_all_features_data, get_highlight_data

# Load listing data, please run this at the very beginning to gather necessary data for running the project codes. 
all_features = get_original_all_features_data()

```

### Running Visualization
```python
# Generate ELO plots
python benchmark/elo_plot.py

# Generate win rate plots
python benchmark/win_rate_plot.py
```

## Requirements

The main dependencies include:
- PyTorch
- Transformers
- OpenAI
- Datasets
- Pandas
- NumPy
- Matplotlib
- And many others (see `requirements.txt` for complete list)

## Contributing

This codebase is for research purposes. If you find issues or have suggestions, please open an issue or contact the authors.

## Contact

For questions about this research, please refer to the paper or contact the authors directly.
