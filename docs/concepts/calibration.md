# Calibration in Evaluation

**Calibration** is the primary measure of forecast quality in `xrtm-eval`. Unlike accuracy (which is binary), calibration measures "how much you know about what you don't know."

## The Brier Score

We use the decomposed Brier Score as our north star metric.

$$ Brier = \frac{1}{N} \sum (f_t - o_t)^2 $$

Where $f_t$ is the forecast probability and $o_t$ is the outcome (0 or 1).

### Decomposition
`xrtm-eval` breaks this down into:
1.  **Reliability**: How close are the probabilities to the true frequency? (e.g., when you say 70%, does it happen 70% of the time?)
2.  **Resolution**: How distinct are your predictions? (Are you always guessing 50/50, or are you bold?)
3.  **Uncertainty**: The inherent noise of the event class.

## Expected Calibration Error (ECE)

ECE is the weighted average of the gap between confidence and accuracy bins.

$$ ECE = \sum_{k=1}^K \frac{|B_k|}{N} | acc(B_k) - conf(B_k) | $$

Evaluators in this package MUST derive from `Evaluator` protocol and support these metrics.
