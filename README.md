# F1 Pit Stop Strategy Search

Models Formula One pit stop strategy as a deterministic sequential decision problem and solves it using Levin Tree Search algorithm.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first load the dashboard automatically fetches race data from the OpenF1 API, runs the data pipeline, fits the policy models, and executes the search. Subsequent loads use cached data.

## Method

The search state is `(lap, compound, tire_age)`. At each step the algorithm can continue on the current tires or pit to any of the three dry compounds, with the goal of minimizing total race time over 58 laps while satisfying the F1 mandatory two-compound rule.

Levin Tree Search assigns each node $n$ a cumulative probability $$p(n) = p(parent) × \pi (action)$$ and expands nodes in order of increasing Levin cost $$\log (depth) − \log (p(n))$$ This biases the search toward strategies that are both short and reached via high-probability actions(ie. trategies that look like real F1 driving).

The action probabilities $\pi$ come from two logistic regression models trained on stint data from all drivers in the session:

- Pit decision model — predicts $P(pit)$ vs $P(continue)$ given lap number, laps remaining, tire age, compound, and expected lap time from the degradation model.
- Compound choice model — predicts which compound is chosen given a pit, as a multiclass classifier over `SOFT` / `MEDIUM` / `HARD`.

A pruning threshold is swept from $0.0001$ to $0.5$ to find the highest value that cuts node expansions without changing the optimal solution.

## Data

Race data comes from the 2024 Australian Grand Prix via the [OpenF1 API](https://openf1.org). Four endpoints are used: `/laps`, `/stints`, `/pit`, and `/weather`. Data is cached to `data/raw/` after the first fetch.
