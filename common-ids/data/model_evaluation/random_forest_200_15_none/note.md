## Overall Performance

Accuracy: 99.79% → Model performs extremely well on the entire dataset.

Weighted Avg F1-score: 99.77% → Weighted average F1-score shows good balance between precision and recall.

Macro Avg F1-score: 85.25% → Average F1-score across all classes regardless of sample size. This indicates performance imbalance between classes.

## Performance by Class

High-performing classes:

- Attack types like DDoS, DoS Hulk, FTP-Patator, PortScan, SSH-Patator all have F1-scores > 99%, showing good classification of common attacks.

- BENIGN has near-perfect accuracy (99.93%), indicating almost no classification errors with normal traffic.

Low-performing classes:

- Bot: Only 32.71% Recall, meaning the model misses many actual bot samples, despite 97.96% precision.

- XSS: Extremely low F1-score (8.78%), recall only 4.59%, indicating the model almost fails to detect XSS attack samples.

- Brute Force: 92.26% Recall but only 69.27% Precision, showing many false positives.

## Conclusion

The model performs well on common attacks but struggles with detecting rare attack types like Bot and XSS.

High precision but low recall in some classes → Need to improve detection capability for less common attacks.
