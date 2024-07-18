from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical


def knn(X_train, y_train):
    knn_model = KNeighborsClassifier()
    # Define the parameter grid for grid search
    param_grid = {
        "n_neighbors": [3, 5, 7, 10],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(knn_model, param_grid, cv=3, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_knn_model = grid_search.best_estimator_
    return best_knn_model


def logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()

    # Define the parameter grid for grid search
    param_grid = {"penalty": ["l2"], "C": [0.01, 0.1, 1.0, 10]}

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(lr_model, param_grid, cv=3, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_lr_model = grid_search.best_estimator_
    return best_lr_model


def svm(X_train, y_train):
    svm_model = SVC(probability=True)
    # Define the parameter grid for grid search
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10],
        "kernel": ["linear", "rbf", "sigmoid", "poly"],
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(svm_model, param_grid, cv=3, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_svm_model = grid_search.best_estimator_
    return best_svm_model


def naive_bayes(X_train, y_train):
    nb_model = GaussianNB()
    # No hyperparameters to tune for Naive Bayes
    nb_model.fit(X_train, y_train)
    return nb_model


def random_forest(X_train, y_train):
    rf_model = RandomForestClassifier()
    # Define the parameter grid for grid search
    param_grid = {"n_estimators": [10, 100, 200], "max_depth": [None, 10, 50]}
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_rf_model = grid_search.best_estimator_
    return best_rf_model


def binary_neural_network(X_train, y_train, net_type):
    if net_type == "snn":

        def create_model():
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    64, activation="relu", input_shape=(X_train.shape[1],)
                )
            )
            model.add(keras.layers.Dense(32, activation="relu"))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            return model

        model = KerasClassifier(model=create_model, verbose=0)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [10, 30, 50, 100]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_snn_model = grid_search.best_estimator_
        return best_snn_model

    elif net_type == "dnn":

        def create_model():
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    128, activation="relu", input_shape=(X_train.shape[1],)
                )
            )
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation="relu"))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation="relu"))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            return model

        model = KerasClassifier(model=create_model, verbose=0)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [10, 30, 50, 100]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_dnn_model = grid_search.best_estimator_
        return best_dnn_model


def multiclass_neural_network(X_train, y_train, net_type):
    y_train = to_categorical(y_train, 3)
    if net_type == "snn":

        def create_model():
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    64, activation="relu", input_shape=(X_train.shape[1],)
                )
            )
            model.add(keras.layers.Dense(32, activation="relu"))
            model.add(keras.layers.Dense(3, activation="softmax"))
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            return model

        model = KerasClassifier(model=create_model, verbose=0)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [10, 30, 50, 100]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_snn_model = grid_search.best_estimator_
        return best_snn_model

    elif net_type == "dnn":

        def create_model():
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    128, activation="relu", input_shape=(X_train.shape[1],)
                )
            )
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation="relu"))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation="relu"))
            model.add(keras.layers.Dense(3, activation="softmax"))
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            return model

        model = KerasClassifier(model=create_model, verbose=0)
        param_grid = {"batch_size": [32, 64, 128], "epochs": [10, 30, 50, 100]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid_search.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_dnn_model = grid_search.best_estimator_
        return best_dnn_model
