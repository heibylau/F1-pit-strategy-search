# F1 Pit Stop Strategy Search
Models Formula One pit stop strategy as a deterministic sequential decision problem
and solves it using A\* and Levin's Tree Search.

## Setup
Clone this repo and install the dependencies:
```bash
pip install -r requirements.txt
```
After installing the requirements, run the project:
```bash
python main.py
```

## Project Structure
```
.
├── main.py            # Entry point for running the strategy search
├── search.py          # Implements A* and Levin Tree Search
├── F1State.py         # State class used in the problem
├── node.py            # Node class for A* and Levin Tree Search
├── model.py           # Logistic regression models for decision probabilities
├── get_parameters.py  # Retrieves key parameters for the problem
├── race_log.py        # Extracts and generates race logs
├── data/              # Raw and processed data files
└── images/            # Visualizations
```

## Data
This project uses data from the 2024 Australian Grand Prix, obtained via the [OpenF1 API](https://openf1.org).