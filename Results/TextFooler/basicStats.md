ok: [0 1 2]:[ 20 244  75]
0 -> Labels [1 2]:[398  82]
1 -> Labels [0 2]:[120 109]
2 -> Labels [0 1]:[ 34 268]
+===========================================+
|                  Summary                  |
+===========================================+
| Total Attacked Instances:       | 1350    |
| Successful Instances:           | 1011    |
| Attack Success Rate:            | 0.74889 |
| Avg. Running Time:              | 0.31483 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 115.29  |
| Avg. Fluency (ppl):             | 127.17  |
| Avg. Grammatical Errors:        | 22.765  |
| Avg. Levenshtein Edit Distance: | 21.119  |
| Avg. Word Modif. Rate:          | 0.33913 |
+===========================================+
+===========================================+

              precision    recall  f1-score   support

           0       0.11      0.04      0.06       500
           1       0.27      0.52      0.35       473
           2       0.28      0.20      0.23       377

    accuracy                           0.25      1350
   macro avg       0.22      0.25      0.22      1350
weighted avg       0.22      0.25      0.21      1350