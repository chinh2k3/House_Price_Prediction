from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
import numpy as np

def model_train(X_train, y_train_log):
    ridge = Ridge(random_state=42)

    alphas = np.logspace(-3, 3, 30)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    grid = GridSearchCV(ridge, param_grid={'alpha': alphas}, cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train_log)

    best_model = grid.best_estimator_
    best_alpha = grid.best_params_['alpha']
    best_score = grid.best_score_

    return best_model, best_alpha, best_score