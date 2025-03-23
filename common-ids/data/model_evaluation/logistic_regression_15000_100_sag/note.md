## Overall Performance

Accuracy of ~96.47% is quite high, indicating good classification capability of the model.

Macro average F1-score = 0.65 → reflects imbalance between classes, with some classes having very low F1-scores.

Weighted average F1-score = 0.96 → due to some dominant classes (BENIGN, DoS Hulk, DDoS), the average results appear high.

## Performance by Class

BENIGN (normal traffic): Precision ~98.46%, Recall ~97.28%, F1-score ~97.87% → model performs well in detection.

Bot, Brute Force, XSS:

- Bot has 100% Precision but extremely low Recall (0.17%), meaning the model rarely predicts Bot correctly.

- Brute Force and XSS: Precision and Recall = 0 → model fails to detect these attacks.

DDoS & DoS GoldenEye, Hulk, Slowhttptest:

- DDoS: Precision ~98.94%, Recall ~97.11% → good detection.

- DoS GoldenEye: Recall ~86.63%, partially confused with BENIGN.

- DoS Hulk: Recall ~91.19%, room for improvement.

- DoS Slowhttptest: Recall ~87.94%, good but not optimal.

- DoS Slowloris: Recall ~53.48%, high confusion rate.

PortScan:

- Precision ~74.59%, Recall ~98.59% → model catches almost all PortScan but with significant false positives.

FTP-Patator, SSH-Patator:

- FTP-Patator: Precision ~93.28%, Recall ~99.75% → good detection.

- SSH-Patator: Recall only 48.67%, more than half confused with BENIGN.

## Conclusion

The model performs well with common attacks (BENIGN, DoS Hulk, DDoS, PortScan) but needs improvement in detecting rare classes like Bot, Brute Force, and XSS.
