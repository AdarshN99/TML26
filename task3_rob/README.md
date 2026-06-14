## Task 3 - Robustness

### Best Score

Best Score Acheived on Public Leaderboard is **NA** 

### To reproduce the results:
- Execute: `python task_template_ensemble.py`
- Results will be stored in `submissions_ensemble.csv`

### Results for different methods:

| Method                                           | Score            |
|--------------------------------------------------|------------------|
| r-FGSM - resnet34 - 50 epochs                    | No Improvement   |
| r-FGSM - resnet50 - 30 epochs                    | 0.407759         |
| r-FGSM - resnet50 - 50 epochs                    | 0.498578         |
| r-FGSM - resnet34 - 50 epochs - Aug              | 0.505333         |
| r-FGSM - resnet50 - 50 epochs - Aug              | No Improvement   |
| r-FGSM - resnet50 - Mix loss(0.5c + 0.5a)        | No Improvement   |
| r-FGSM - resnet50 - Mix loss(0.3c + 0.7a)        | No Improvement   |
| PGD - resnet50                                   | No Improvement   |
| Claude - resnet50                                | 0.560480         |
| Trades - resnet50                                | No Improvement   |
| Claude + Trades - resnet50                       | No Improvement   |