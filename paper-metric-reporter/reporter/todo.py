"""
+ Read predictions
+ Read actuals
+ Calculate MAPE
+ Add test MAPE values
+ Calculate all combinations of Long-Short portfolio
+ Calculate random portfolio
+ Calculate all portfolio
+ Calculate statistical significance
+ Create the following dataframes
  + Best Mape, Accuracy and F1-score, X* Values that give the highest portfolio return
  + *X is the metric that calculates the ratio of the highest increases (and decrease for shorting) in predictions
    over reality, find a good name for this metric. The name can be "Asset Selection Accuracy".
                       |  MAPE | Accuracy | F1-score |   X  |
  -------------------- +-------+----------+----------+-------
  Random               |    X  |    X     |    X     |   X
  All                  |    X  |    X     |    X     |   X
  LSTM                 |    X  |    X     |    X     |   X
  Connecting the Dots  |    X  |    X     |    X     |   X
  DeepGLO              |    X  |    X     |    X     |   X

  + Portfolio Results
  Market Condition |   Model  | Expected Return | Volatility | Information Ratio | Maximum Drawdown
  -----------------+----------+-----------------+------------+-------------------+-------------------
  Bull             | ctd      |        X        |     X      |        X          |        X
                   | deepglo  |        X        |     X      |        X          |        X
                   | lstm     |        X        |     X      |        X          |        X
                   | random   |        X        |     X      |        X          |        X
                   | all      |        X        |     X      |        X          |        X
  Bear             | ctd      |        X        |     X      |        X          |        X
                   | deepglo  |        X        |     X      |        X          |        X
                   | lstm     |        X        |     X      |        X          |        X
                   | random   |        X        |     X      |        X          |        X
                   | all      |        X        |     X      |        X          |        X

  + Statistical Significance
  Comparison        | T-Statistic | P-Value | Result
  ------------------+-------------+---------+--------
  ctd vs deepglo    |      X      |    X    |   X
  ctd vs lstm       |      X      |    X    |   X
  deepglo vs lstm   |      X      |    X    |   X
  ctd vs random     |      X      |    X    |   X
  ctd vs all        |      X      |    X    |   X
  deepglo vs random |      X      |    X    |   X
  deepglo vs all    |      X      |    X    |   X
  lstm vs random    |      X      |    X    |   X
  lstm vs all       |      X      |    X    |   X
"""
