# Credit-card-fraud

Data set used in this project is downloaded from Kaggle (https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).

This is apply of exploratory data analysis and making different classification and regression models to detect credit card fraud. The data set is unbalanced, so the main criterion of model quality is F1-score (harmonic mean of precission and sensitivity). As you can see in the given documents F1-score is 0.99 even with relatively simple tree. Hence F1-score will be extremly well (0.99) with ensemble methods. I've also tried Random Forest and XGBoosting just to see how fast it is, because of the way XGB is implemented.

Parameters of models were chosen on validation set (somewhere it was cross-validation). Of course, final metrics (accuracy, precission, sensitivity and f1-score) were determined on the testing set.
