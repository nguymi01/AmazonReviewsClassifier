import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import math

# from numba import jit, cuda
# import warnings
# warnings.filterwarnings('ignore')

# @jit
def fine_tune(X_train_processed, X_submission_processed, X_test_processed, Y_train, Y_test):

    best_score = math.inf
    best_submission = None
    best_depth = -1
    best_model = None

    depth_arr = [50]
    estimators_arr = [100, 125, 150]
    features_max = ["sqrt", "log2"]
    criterion_arr = ["gini", "entropy", "log_lostt"]
    # weight_arr = ["balanced", "balanced_subsample"]

    for depth in depth_arr:
        # for estimator in estimators_arr:
        #     for features in features_max:
        #         for weight in weight_arr:
        #             for criterion in criterion_arr:
        model = RandomForestClassifier(max_depth=depth
                                    # n_estimators=estimator,
                                    # criterion=criterion,
                                    # max_features=features,
                                    # class_weight=weight
                                    ).fit(X_train_processed, Y_train)

        # Predict the score using the model
        Y_test_predictions = model.predict(X_test_processed)

        rmse = mean_squared_error(Y_test, Y_test_predictions)**(1/2)

        if rmse < best_score: 
            best_score = rmse
            best_depth = depth
            best_model = model

        print("Depth=" + str(depth) + 
                # " N_estimators=" + str(estimator) +  
                # " Criterion=" + criterion + 
                # " Max_features=" + features + 
                # " Weight=" + weight
                "")
        # Evaluate your model on the testing set
        print("   Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))
        print("   Root Mean Squared Error = ", rmse) 
        print()

    X_submission['Score'] = best_model.predict(X_submission_processed)
    best_submission = X_submission

    return (best_score, best_depth, best_submission)


if __name__ == "__main__":

    # Load files into DataFrames
    X_train = pd.read_csv("./data/X_train.csv")
    X_submission = pd.read_csv("./data/X_test.csv")

    # Split training set into training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(
            X_train.drop(['Score'], axis=1),
            X_train['Score'],
            test_size=1/4.0,
            random_state=0
        )

    # This is where you can do more feature selection
    X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
    X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
    X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])

    # Learn the model
    best_score, best_depth, best_submission = fine_tune(X_train_processed, 
                                                        X_submission_processed, 
                                                        X_test_processed, 
                                                        Y_train, 
                                                        Y_test)

    # Create the submission file
    # submission = X_submission[['Id', 'Score']]

    print("Best Depth =", best_depth)
    print("Best Score =", best_score)

    submission = best_submission[['Id', 'Score']]
    submission.to_csv("./data/randforest_submission.csv", index=False)
