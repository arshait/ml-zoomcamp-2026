# Homework 4 â€” Answers (ML Zoomcamp 2026)

Dataset: `data/course_lead_scoring_2026.csv`
Target: `converted`
Splits: `test_size=0.2`, then `test_size=0.25` on remainder, `random_state=1`.

## Question 1 â€” Best single numerical feature (ROC AUC on train)

| Feature | ROC AUC |
|---------|---------|
| lead_score | **0.788225** |
| interaction_count | 0.764920 |
| number_of_courses_viewed | 0.722989 |
| annual_income | 0.608198 |

**Answer: `lead_score`**

## Question 2 â€” Logistic regression validation ROC AUC

DictVectorizer + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`.

**Validation ROC AUC: 0.732**

**Answer: 0.732**

## Question 3 â€” Threshold with smallest |precision âˆ’ recall|

After excluding thresholds where both precision and recall are zero:

- Best threshold: **0.63** (precision â‰ˆ 0.7328, recall â‰ˆ 0.7253)

**Answer: 0.63**

Plot: `plots/precision_recall_vs_threshold.png`

## Question 4 â€” Maximum F1 threshold

**Answer: 0.41** (max F1 â‰ˆ 0.7576)

## Question 5 â€” 5-fold CV standard deviation (C=1)

Fold AUCs: `[0.731, 0.730, 0.736, 0.741, 0.721]`
Mean AUC: **0.732**
Std AUC: **0.007** (exact: 0.006853)

**Answer: 0.007**

## Question 6 â€” Best regularization C

| C | Mean AUC (3 dp) | Std (3 dp) |
|---|-----------------|------------|
| 0.000001 | 0.617 | 0.019 |
| **0.001** | **0.749** | **0.010** |
| 1 | 0.732 | 0.007 |

**Answer: C = 0.001** (highest mean AUC; tie-break not needed)
