import pandas as pd

from sajacaros_learn.Regression import NpLinearRegression


def load_mission_dataset():
    return pd.read_csv('data/mission_train.csv'), pd.read_csv('data/mission_test.csv')


def energy_predict():
    train_df, test_df = load_mission_dataset()

    np_linear_reg = NpLinearRegression()
    # np_linear_reg.fit(X, y, lr=0.1)
    # y_hat = np_linear_reg.predict(t)
    # error_report(y, y_hat)


if __name__ == '__main__':
    energy_predict()
