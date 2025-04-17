import pandas as pd
import textstat


def processDate(df, date_col):
    """
    Create year, month, day of week, and hour columns in a dataframe based on a date column.

    :param df: (pandas DataFrame) dataframe containing a date column
    :param date_col: (string) column name of the date column
    """
    assert date_col in df.columns

    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])

    df['Factor.year'] = df[date_col].dt.year.astype(float)
    df['Factor.month'] = df[date_col].dt.month.astype(float)
    df['Factor.day_of_week'] = df[date_col].dt.day_of_week.astype(float)
    df['Factor.hour'] = df[date_col].dt.hour.astype(float)

    return df


def createConfusionMatrix(df, label_col, prediction_col, positive_value):
    """
    Create columns with boolean indicators of whether the prediction is a True Positive, False Positive, True Negative, or False Negative.

    :param df: (panda dataframe) dataframe containing the predicted and true values
    :param label_col: (string) column name of the column containing the ground truth values
    :param prediction_col: (string) column name of the column containing the predicted values
    :param positive_value: (string) the value of the true and predicted variables corresponding to a positive prediction
    """
    assert label_col in df.columns
    assert prediction_col in df.columns

    df = df.copy()

    df['Outcome.True_Positive'] = df.apply(
        lambda x: 1 if x[label_col] == positive_value and x[prediction_col] == positive_value else 0, axis=1
    ).astype(int)
    df['Outcome.False_Negative'] = df.apply(
        lambda x: 1 if x[label_col] == positive_value and x[prediction_col] != positive_value else 0, axis=1
    ).astype(int)
    df['Outcome.True_Negative'] = df.apply(
        lambda x: 1 if x[label_col] != positive_value and x[prediction_col] != positive_value else 0, axis=1
    ).astype(int)
    df['Outcome.False_Positive'] = df.apply(
        lambda x: 1 if x[label_col] != positive_value and x[prediction_col] == positive_value else 0, axis=1
    ).astype(int)

    return df


def createPredictionCols(df, label_col, prediction_col, positive_value, confidence_col=None):
    """
    Create columns relatead to the model predictions.

    :param df: (panda dataframe) dataframe containing the predicted and true values
    :param label_col: (string) column name of the column containing the ground truth values
    :param prediction_col: (string) column name of the column containing the predicted values
    :param positive_value: (string) the value of the true and predicted variables corresponding to a positive prediction
    :param confidence_col: (string), optional, column name of column containing the prediction confidence.
    """
    assert label_col in df.columns
    assert prediction_col in df.columns

    df = df.copy()

    df['Outcome.Correctness'] = (df[label_col] == df[prediction_col]).astype(int)

    if not isinstance(positive_value, list):
        df['Outcome.Predicts_Positive'] = (df[prediction_col] == positive_value).astype(int)
        df['Outcome.Predicts_Negative'] = (df[prediction_col] != positive_value).astype(int)

    else:
        for col in positive_value:
            df[f'Outcome.Predicts_{col}'] = (df[prediction_col] == col).astype(int)

        df['Outcome.Predicts_Other'] = (~df[prediction_col].isin(positive_value)).astype(int)

    if confidence_col is not None:
        df['Outcome.Prediction_Confidence'] = df[confidence_col].copy()
    elif not isinstance(positive_value, list) and positive_value in df.columns:
        df['Outcome.Prediction_Confidence'] = df.apply(lambda x: x[x[prediction_col]], axis=1)

    return df


def processCategoricalCols(df, categorical_cols):
    """
    One hot encode categorical columns in the dataframe.

    :param df: (pandas DataFrame) a dataframe containing column(s) with categorical variables
    :param categorical_cols: (list) a list of the column names corresponding to categorical variables
    """
    df = df.copy()

    for col in categorical_cols:

        assert col in df.columns

        one_hot = pd.get_dummies(df[col])
        one_hot.columns = [f'Factor.{col}-{c}' for c in one_hot.columns]
        df = pd.concat([df, one_hot.astype(int)], axis=1)

    return df


def createTextFactors(df, text_col):
    """
    Generate features based on the input text (e.g. number of tokens, readability, etc.).

    :param df: (pandas DataFrame) a dataframe containing a text column
    :param text_col: (string) the name of the text column
    """
    assert text_col in df.columns

    df = df.copy()

    df["Factor.numTokens"] = df['text'].apply(lambda x: len(x.split(" ")))
    df["Factor.Readability"] = df['text'].apply(lambda x: textstat.gunning_fog(x))
    df["Factor.Difficult_Words"] = df['text'].apply(lambda x: textstat.difficult_words(x)) / df['Factor.numTokens']

    return df
