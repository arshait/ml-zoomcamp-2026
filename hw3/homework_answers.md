# ML Zoomcamp 2026 — Homework 3 Answers

| Question | Topic | Verified Answer |
|---|---|---|
| Q1 | Industry mode | **technology** |
| Q2 | Biggest correlation | **interaction_count and lead_score** |
| Q3 | Mutual information | **lead_source** |
| Q4 | Logistic regression accuracy | **0.65** |
| Q5 | Feature elimination | **number_of_courses_viewed** |
| Q6 | Best regularization C | **0.001** |

## Numerical evidence

### Q1 — Industry mode

After filling categorical missing values with `"NA"`:

- `technology`: 1173
- `retail`: 925
- `healthcare`: 840
- `NA`: 240 (missing imputed)

Mode: **technology**

### Q2 — Correlation (Pearson, among listed pairs only)

| Pair | Correlation |
|---|---|
| interaction_count ↔ lead_score | **0.915746** |
| number_of_courses_viewed ↔ lead_score | 0.757204 |
| number_of_courses_viewed ↔ interaction_count | 0.721609 |
| annual_income ↔ interaction_count | 0.122842 |

Largest: **interaction_count and lead_score**

### Q3 — Mutual information (training set, `round(score, 2)`)

| Feature | MI (rounded) |
|---|---|
| **lead_source** | **0.03** |
| employment_status | 0.02 |
| industry | 0.0 |
| location | 0.0 |

Highest among options: **lead_source**

### Q4 — Logistic regression (validation)

- Raw accuracy: `0.645`
- Rounded to 2 decimals: **0.65**

Model: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)` with `DictVectorizer(sparse=False)`.

### Q5 — Feature elimination (unrounded baseline)

Baseline validation accuracy: `0.645`

| Removed feature | Accuracy without | Difference (baseline − reduced) |
|---|---|---|
| lead_source | 0.642 | 0.003 |
| **number_of_courses_viewed** | 0.643 | **0.002** |
| interaction_count | 0.601 | 0.044 |

Smallest difference: **number_of_courses_viewed**

### Q6 — Regularization (validation, `round(accuracy, 3)`)

| C | Rounded accuracy |
|---|---|
| 0.000001 | 0.598 |
| 0.00001 | 0.598 |
| 0.0001 | 0.613 |
| **0.001** | **0.645** |

Best (unique maximum): **0.001**

## Submission URL

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03

## Submission Checklist

- [x] All six answers verified (Docker Python 3.12, two runs — identical results)
- [x] Notebook generated (`hw3_classification_2026.ipynb`)
- [x] Python script runs successfully in Docker (`hw3_solution.py`)
- [ ] Answers copied to the submission form
- [ ] Homework URL added if publicly available
- [ ] Submission saved before deadline (October 13, 2026, 04:30 your timezone)
