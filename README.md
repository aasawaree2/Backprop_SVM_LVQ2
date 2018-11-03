# Backprop_SVM_LVQ2
Compare the performance (quality and computational effort) of LVQ2, Backprop, and SVM on a classification problem

SVM :
Accuracy - 94.02
Standard_Deviation - 0.97
Variance - 0.95
Total execution time - 0.2039
Backpropagation :
Accuracy - 93.26
Standard_Deviation - 0.44
Variance - 0.19
Total execution time - 1.6242
LVQ2 :
Accuracy - 91.83
Standard_Deviation - 1.10
Variance - 1.21
Total execution time - 0.6684

Student’s T-Test :-

The t-test is a ratio. The numerator is the difference of the two means and the denominator is a measure of variability. This is similar to the signal to noise ratio.

Here, I have taken α = 0.05 and the degree of freedom value would be calculated using the formula : n 1 + n 2 - 2 = 16 + 16 - 2 = 30, since we run each algorithm for 16 iterations. The critical value for all the three algorithm would be the same i.e. ± t 0.95 (30) = ± 2.042 as each algorithm runs for 16 iterations. This value is obtained by referencing the critical values of Student’s t distribution table.

Testing Hypothesis :-

Reject H0 , if t-value is not in the critical value range
Fail to Reject H0 , if t-value is in the critical value range

SVM vs Backpropagation :-
t =  ± 2.847 
Null Hypothesis - H0 : μSVM = μBackpropagation
Alternate Hypothesis - H1 : μSVM =/ μBackpropagation
Here, t = ± 2.847, which is greater than the critical value range. Hence, the null hypothesis H0 is rejected. This concludes that accuracy was improved.

SVM vs LVQ2 :-
t =  ± 5.960 
Null Hypothesis - H0 : μSVM = μLV Q2
Alternate Hypothesis - H1 : μSVM =/ μLV Q2
Here, t = ± 5.960, which is greater than the critical value range. Hence, the null hypothesis H0 is rejected. This concludes that accuracy was improved.

Backpropagation vs LVQ2 :-
t =  ± 4.834 
Null Hypothesis - H0 : μBackpropagation = μLV Q2
Alternate Hypothesis - H1 : μBackpropagation =/ μLV Q2
Here, t = ± 4.834, which is greater than the critical value range. Hence, the null hypothesis H0 is rejected. This concludes that accuracy was improved.

The above result may vary depending upon the chosen dataset and how each model was initialized for training.

References :-
1. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
2. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
3. http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
4. http://neupy.com/apidocs/neupy.algorithms.competitive.lvq.html
5. https://www.socialresearchmethods.net/kb/stat_t.php
6. https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
7. https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.var.html
8. https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html
9. https://www.youtube.com/watch?v=pTmLQvMM-1M&t=323s
10. https://www.youtube.com/watch?v=BPbHujvA9UU
