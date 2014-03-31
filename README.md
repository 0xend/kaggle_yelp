Trying to predict useful votes on a Yelp review based on user/biz/text. (Kaggle challenge: http://www.kaggle.com/c/yelp-recruiting/rules)

Files:

- TrainerModel: Sets the interface and implements the common methods for individual trainers.
- ReviewTrainer: Uses Ridge regression to try to predict the number of useful votes based on 
the text of the review. The choice of Ridge over non-linear models is the fact that the data
is already in a very high dimension (unigrams are features). Before fitting the model, it selects
the most relevant K features (15000 in this case).
- BusinessTrainer: It predicts the number of useful votes based on certain features of the business.
Since the number of examples is not too big and m << n (m = number of features, n = examples), it uses
an SVM regressor with an RBF kernel. The choice was based on the fact that the data is likely
not linearly separable and we can still use an SVM since we don't have that many examples (if it
was bigger, it might become too computationally expensive.
-UserTrainer: In this case, we choose again a non-linear model. However, the number of examples make it
too expensive to compute an SVM with an rbf/other non-linear kernel. AdaBoost and Random Forests were tried.
Based on the performance, I ended up keeping Random Forest. If a bigger number of examples is available,
this method will likely still be useful since it's parallelizable.

*For business and user, the number of useful votes (label) is given by the average of all
examples seen for such biz/user. Another approach that is worth trying is instead of constructing just
using y(X_i) = avg(y_i), build n examples (given that we have n labels for example X_i. For instance:
if we have for X_i, the labels [1, 0, 2], instead of having y(X_i) = 1, build three examples (one
for each label).


To run it:
./trainer.py <ratio>

*Ratio = # training samples / # test samples

TODO:

- Incorporate check-ins info to the models.
- Use learning with experts advice such that each model mentioned above represents one of the experts.
I anticipate the performance should be at least as good as that of the best model if not better since
each of them is using different attributes which means the information they provide can add up 
together (the overlapping of information provided by the models shouldn't be 'too high').


