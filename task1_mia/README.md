## Task 1 - Privacy: Membership Inference Attack

### Best Score

Best Score Acheived on Public Leaderboard is **0.060290** with LiRA (32 Shadow models with 6 epochs) <br>

### To reproduce the results:
- run the command: `python task_template_lira.py`
- Results will be stored in `submissions.csv`
- To submit to the leaderboard server, add API_KEY and run `python submission.py`

### Results for different methods:

| Method                              | Score       |
|-------------------------------------|------------|
| Maximum Confidence Score            | 0.051502   |
| Negative Loss                       | 0.051911   |
| LiRA (5 epochs, 16 models)          | 0.059881   |
| LiRA (6 epochs, 32 models)          | 0.060290   |
| LiRA (10 epochs, 64 models)         | No Improvement |
| LiRA (20 epochs, 128 models)        | No Improvement |
| RMIA (32 models)                    | No Improvement |
| RMIA (64 models)                    | No Improvement |
