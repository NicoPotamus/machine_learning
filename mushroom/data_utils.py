import pandas as pd
import numpy as np


def get_clean_X_y(csv_path='mushrooms_clean.csv', source_path='mushrooms.csv'):
    """Return (X, y) where X is the dummified feature matrix and y is the label series.

    Behavior:
    - If csv_path exists, read and return X, y from that cleaned CSV.
    - Otherwise, read source_path, drop rows with missing values, create dummies for non-numeric
      columns (excluding 'class'), and return X, y.
    """
    try:
        df = pd.read_csv(csv_path)
        # assume last column is 'class' or has a 'class' column
        if 'class' in df.columns:
            y = df['class']
            X = df.drop(columns=['class'])
        else:
            # fallback: assume last column is label
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        return X, y
    except FileNotFoundError:
        # build from source
        df_src = pd.read_csv(source_path)
        df_src = df_src.replace('?', np.nan)
        df_src = df_src.dropna().reset_index(drop=True)
        # tidy column names
        df_src.columns = [c.replace('-', '_') for c in df_src.columns]
        y = df_src['class']
        non_numeric = df_src.select_dtypes(exclude=['number']).columns.tolist()
        non_numeric = [c for c in non_numeric if c != 'class']
        numeric_cols = df_src.select_dtypes(include=['number']).columns.tolist()
        if len(non_numeric) > 0:
            X_cat = pd.get_dummies(df_src[non_numeric], drop_first=False)
        else:
            X_cat = pd.DataFrame(index=df_src.index)
        if len(numeric_cols) > 0:
            X_num = df_src[numeric_cols].reset_index(drop=True)
            X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
        else:
            X = X_cat.reset_index(drop=True)
        # Optionally save cleaned CSV for faster future loads
        df_clean = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        try:
            df_clean.to_csv(csv_path, index=False)
        except Exception:
            pass
        return X, y
