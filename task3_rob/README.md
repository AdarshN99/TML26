## Task 3 - Robustness

### Best Score

Best Score Acheived on Public Leaderboard is **0.568144** 

### To reproduce the results:
- Execute: `python best.py`

### Results for different methods:

| Method                                                                                | Score            |
|---------------------------------------------------------------------------------------|------------------|
| r-FGSM - resnet50 - 30 epochs                                                         | 0.407759         |
| r-FGSM - resnet50 - 50 epochs                                                         | 0.498578         |
| PGD - resnet50                                                                        | No Improvement   |
| **Curriculum Adversarial Training with FGSM and PGD - resnet50**                      | **0.568144**     |
| TRADES with CutMix                                                                    | No Improvement   |