# Four4All Cross-Database Results

This file contains all the classification report for the Four4All model across different datasets.

The datasets that were tested:
 - ✅ FER2013
 - ✅ RAF-DB
 - ✅ Four4All
 - ✅ CK+
 - ⏳ FERPlus (Coming soon)
 - ⏳ AffectNet (Coming soon)

## FER2013 Classification Report	

|           | Precision |	Recall |	f1-score |	Support|
|-----------|-----------|--------|-----------|---------|
| happiness |   0.75    |	 0.85  |	 0.79    |	 100   |
| surprise  |	  0.84  	|  0.81  |   0.82	   |   100   |
| sadness   |	  0.49    |	 0.57  |	 0.53    |	 100   |
| anger	    |   0.49  	|  0.58	 |   0.53	   |   100   |
| disgust   |	  0.94    |	 0.58  |	 0.72    |	 100   |
| fear	    |   0.54	  |  0.5	 |   0.52	   |   100   |	
|           |           |        |           |         |
| accuracy  |           |        |	 0.65    |	 600   |
| macro avg |	  0.67    |	 0.65  |	 0.65    |	 600   |
| weighted avg |	0.67  |	 0.65  |	 0.65	   |   600   |



## RAF-DB Classification Report

|           | Precision |	Recall |	f1-score |	Support |
|-----------|-----------|--------|-----------|----------|
| happiness |   0.81    |	 0.91  |	0.85     |	 100    |
| surprise  |	  0.79  	|  0.84  |  0.82	   |   100    |
| sadness   |	  0.88    |	 0.84  |	0.86     |	 100    |
| anger	    |   0.86  	|  0.83	 |  0.85	   |   100    |
| disgust   |	  0.84    |	 0.80  |	0.82     |	 100    |
| fear	    |   0.88	  |  0.82	 |  0.85	   |   100    |	
|           |           |        |           |          |
| accuracy  |           |        |	0.84     |	 574    |
|macro avg  |	  0.85    |	 0.84  |	0.84     |	 574    |
| weighted avg |	0.84  |	 0.84  |	0.84	   |   574    |


## Four4All Classification Report

|           | Precision |	Recall |	f1-score | Support |
|-----------|-----------|--------|-----------|---------|
| happiness |   0.95    |	 0.87  |	 0.91    |	 150   |
| surprise  |	  0.92  	|  0.88  |   0.90	   |   150   |
| sadness   |	  0.96    |	 0.88  |	 0.92    |	 150   |
| anger	    |   0.89  	|  0.94	 |   0.91	   |   150   |
| disgust   |	  0.87    |	 0.91  |	 0.89    |	 150   |
| fear	    |   0.86	  |  0.95	 |   0.90	   |   150   |	
|           |           |        |           |         |
| accuracy  |           |        |	 0.90    |	 900   |
| macro avg |	  0.91    |	 0.90  |	 0.90    |	 900   |
| weighted avg |	0.91  |	 0.90  |	 0.90	   |   900   |


## CK+ Classification Report

|              | Precision |	Recall |	 f1-score | Support |
|--------------|-----------|--------|-----------|---------|
|    happiness |   1.00    |	 1.00  |	  1.00    |	   75   |
|    surprise  |	  1.00  	 |  1.00  |   1.00	   |    75   |
|    sadness   |	  1.00    |	 1.00  |	  1.00    |	   75   |
|    anger	    |   1.00  	 |  1.00	 |   1.00    |    75   |
|    disgust   |	  1.00    |	 1.00  |	  1.00    |	   75   |
|     fear	    |   1.00	   |  1.00	 |   1.00	   |    75   |	
|              |           |        |           |         |
|   accuracy   |           |        |	  1.00    |	  450   |
|   macro avg  |	  1.00    |	 1.00  |	  1.00    |	  450   |
| weighted avg |	  1.00    |	 1.00  |	  1.00	   |   450   |
