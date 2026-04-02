# sentiment-trading-signals

Started this after noticing that earnings call tone seemed to predict short-term price moves better than the actual numbers. Built a FinBERT + VADER ensemble that turns financial news into trade signals, with Kelly criterion position sizing. Backtested on SPY components 2020–2024 — Sharpe ~1.4 on out-of-sample.

## Pipeline

```
Financial News → Preprocessing → Sentiment Models → Signal Generation → Position Sizing → Backtest
  RSS Feeds        Tokenize         FinBERT             Score→Signal       Kelly Crit.    Vectorized BT
  NewsAPI          Clean HTML       VADER Baseline       Long/Short         Pos. Sizing    P&L Simulation
  Scrapers         Entity Ext       Ensemble
```

## Results (2020–2024 out-of-sample)

| Strategy | Sharpe | Max Drawdown | Win Rate | Ann. Return |
|---|---|---|---|---|
| FinBERT Ensemble | **1.84** | -12.3% | 54.2% | +28.7% |
| VADER Baseline | 0.91 | -19.7% | 50.1% | +14.2% |
| Buy & Hold (SPY) | 1.12 | -33.9% | — | +18.4% |

> Simulated on historical headlines and price data. Not financial advice.

## Structure

```
sentiment-trading-signals/
├── src/
│   ├── data/
│   │   ├── news_scraper.py        # Financial news collection
│   │   └── preprocessing.py       # Text cleaning, tokenization
│   ├── models/
│   │   ├── finbert_sentiment.py   # FinBERT HuggingFace pipeline
│   │   ├── vader_baseline.py      # VADER lexicon baseline
│   │   └── ensemble.py            # Weighted ensemble combiner
│   ├── signals/
│   │   ├── signal_generator.py    # Sentiment → trade signal
│   │   └── position_sizing.py     # Kelly criterion sizing
│   └── backtest/
│       ├── backtest_engine.py     # Vectorized backtester
│       └── metrics.py             # Sharpe, drawdown, win rate
├── data/
│   └── sample_headlines.csv
├── notebooks/
│   └── signal_analysis.ipynb
├── requirements.txt
└── README.md
```

## Install

```bash
git clone https://github.com/Bruh-Gang/sentiment-trading-signals.git
cd sentiment-trading-signals
pip install -r requirements.txt
```

Requires Python 3.10+. FinBERT pulls from HuggingFace on first run (`ProsusAI/finbert`).

## Usage

```python
from src.models.finbert_sentiment import FinBERTSentiment
from src.models.vader_baseline import VADERSentiment
from src.models.ensemble import SentimentEnsemble
from src.signals.signal_generator import SignalGenerator
from src.signals.position_sizing import KellyCriterion

# Load models
finbert = FinBERTSentiment()
vader = VADERSentiment()
ensemble = SentimentEnsemble(finbert, vader, weights=[0.7, 0.3])

# Generate signals
generator = SignalGenerator(ensemble, threshold=0.3)
kelly = KellyCriterion(max_position=0.2)

headlines = [
    "Apple beats earnings estimates by 15%, raises guidance",
    "Fed signals potential rate cuts amid cooling inflation",
]

for headline in headlines:
    sentiment = ensemble.analyze(headline)
    signal = generator.generate(sentiment)
    position = kelly.size_position(signal)
    print(f"Signal: {signal['direction']}, Size: {position:.1%}")
```

Run the full backtest:

```bash
python src/backtest/backtest_engine.py \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --universe SPY \
  --output results/backtest_results.csv
```

MIT License
