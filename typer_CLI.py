import typer
from joblib import dump, load
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train_model(model_name: str, db_url: str = "sqlite:///pizza_sales.db"):
    """
    Train a model based on pre-processed data from a database and save it to a file.
    """
    try:
        engine = create_engine(db_url)
        df = pd.read_sql('SELECT * FROM full_sales_data', engine)
    except Exception as e:
        typer.echo(f"Failed to load data from database: {e}")
        raise typer.Exit(code=1)

    X = df.drop(['order_date','pizza_name'], axis=1)
    y = df['quantity']

    # Split the data
    test_index = df["order_date"].iloc[-1] - pd.DateOffset(months=6)
    X_train, X_test = X.loc[df['order_date'] < test_index], X.loc[df['order_date'] >= test_index]
    y_train, y_test = y.loc[df['order_date'] < test_index], y.loc[df['order_date'] >= test_index]
    test_dates = df.loc[df['order_date'] >= test_index, 'order_date']

    # Model training and validation
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': np.arange(50, 250, 10),
        'max_depth': np.arange(5, 20, 2)
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=15,
                                    cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    
    # Model evaluation
    def predict_next_quantity(X, model, n_periods):
        X_pred = X.copy()
        y_pred = np.zeros(n_periods)
        for i in range(n_periods):
            X_pred = pd.concat([X_pred, X_pred.iloc[-1:, :]], axis=0, ignore_index=True)
            y_pred[i] = model.predict(X_pred.iloc[-1:])
            X_pred['quantity_lag1'].iloc[-1] = y_pred[i]
            X_pred['quantity_lag2'].iloc[-1] = X_pred['quantity_lag1'].iloc[-1]
            X_pred['quantity_lag3'].iloc[-1] = X_pred['quantity_lag2'].iloc[-1]
            X_pred['quantity_lag4'].iloc[-1] = X_pred['quantity_lag3'].iloc[-1]
            X_pred['quantity_lag5'].iloc[-1] = X_pred['quantity_lag4'].iloc[-1]
            X_pred['quantity_lag6'].iloc[-1] = X_pred['quantity_lag5'].iloc[-1]
            X_pred['quantity_lag7'].iloc[-1] = X_pred['quantity_lag6'].iloc[-1]
            X_pred['quantity_lag8'].iloc[-1] = X_pred['quantity_lag7'].iloc[-1]
            X_pred['quantity_lag9'].iloc[-1] = X_pred['quantity_lag8'].iloc[-1]
            X_pred['quantity_lag10'].iloc[-1] = X_pred['quantity_lag9'].iloc[-1]
            X_pred['quantity_lag11'].iloc[-1] = X_pred['quantity_lag10'].iloc[-1]
            X_pred['quantity_lag12'].iloc[-1] = X_pred['quantity_lag11'].iloc[-1]
            X_pred['quantity_lag13'].iloc[-1] = X_pred['quantity_lag12'].iloc[-1]
            X_pred['quantity_lag14'].iloc[-1] = X_pred['quantity_lag13'].iloc[-1]
        return X_pred, y_pred
    model = grid_search.best_estimator_
    X_pred, y_pred = predict_next_quantity(X_train, model, n_periods=30)
    
    # Save the model to a file
    dump(model, f"{model_name}.joblib")
    typer.echo(f"Model {model_name} trained and saved successfully.")




@app.command()
def predict_and_save(month: int, day: int, db_url: str = "sqlite:///pizza_sales.db"):
    engine = create_engine(db_url)

    # Split the data
    initial_date = datetime(2015, month, day).strftime('%Y-%m-%d')
    test_index = initial_date
    X_train, X_test = X.loc[df['order_date'] < test_index], X.loc[df['order_date'] >= test_index]
    y_train, y_test = y.loc[df['order_date'] < test_index], y.loc[df['order_date'] >= test_index]
    test_dates = df.loc[df['order_date'] >= test_index, 'order_date']
    
    try:
        engine = create_engine(db_url)
        df = pd.read_sql('SELECT * FROM full_sales_data', engine)
    except Exception as e:
        typer.echo(f"Failed to load data from database: {e}")
        raise typer.Exit(code=1)

    X = df.drop(['order_date','pizza_name'], axis=1)
    y = df['quantity']

    # Split the data
    initial_date = datetime(2015, month, day).strftime('%Y-%m-%d')
    test_index = initial_date
    X_train, X_test = X.loc[df['order_date'] < test_index], X.loc[df['order_date'] >= test_index]
    y_train, y_test = y.loc[df['order_date'] < test_index], y.loc[df['order_date'] >= test_index]
    test_dates = df.loc[df['order_date'] >= test_index, 'order_date']

    # Model training and validation
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': np.arange(50, 250, 10),
        'max_depth': np.arange(5, 20, 2)
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=15,
                                    cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    
    # Model evaluation
    def predict_next_quantity(X, model, n_periods):
        X_pred = X.copy()
        y_pred = np.zeros(n_periods)
        for i in range(n_periods):
            X_pred = pd.concat([X_pred, X_pred.iloc[-1:, :]], axis=0, ignore_index=True)
            y_pred[i] = model.predict(X_pred.iloc[-1:])
            X_pred['quantity_lag1'].iloc[-1] = y_pred[i]
            X_pred['quantity_lag2'].iloc[-1] = X_pred['quantity_lag1'].iloc[-1]
            X_pred['quantity_lag3'].iloc[-1] = X_pred['quantity_lag2'].iloc[-1]
            X_pred['quantity_lag4'].iloc[-1] = X_pred['quantity_lag3'].iloc[-1]
            X_pred['quantity_lag5'].iloc[-1] = X_pred['quantity_lag4'].iloc[-1]
            X_pred['quantity_lag6'].iloc[-1] = X_pred['quantity_lag5'].iloc[-1]
            X_pred['quantity_lag7'].iloc[-1] = X_pred['quantity_lag6'].iloc[-1]
            X_pred['quantity_lag8'].iloc[-1] = X_pred['quantity_lag7'].iloc[-1]
            X_pred['quantity_lag9'].iloc[-1] = X_pred['quantity_lag8'].iloc[-1]
            X_pred['quantity_lag10'].iloc[-1] = X_pred['quantity_lag9'].iloc[-1]
            X_pred['quantity_lag11'].iloc[-1] = X_pred['quantity_lag10'].iloc[-1]
            X_pred['quantity_lag12'].iloc[-1] = X_pred['quantity_lag11'].iloc[-1]
            X_pred['quantity_lag13'].iloc[-1] = X_pred['quantity_lag12'].iloc[-1]
            X_pred['quantity_lag14'].iloc[-1] = X_pred['quantity_lag13'].iloc[-1]
        return X_pred, y_pred
    model = grid_search.best_estimator_
    X_pred, y_pred = predict_next_quantity(X_train, model, n_periods=30)

    # Concatenate the predictions to the original dataframe
    y_pred = pd.concat([y_train, pd.Series(y_pred)], axis=0)
    y_true = pd.concat([y_train, y_test.iloc[0:30]], axis=0)
    time = pd.concat([df.loc[df['order_date'] < test_index, 'order_date'], test_dates.iloc[0:30]], axis=0)


    # Save y_pred, y_true, and time to a new SQLite table called "final_predictions" inside pizza_sales.db
    final_predictions = pd.DataFrame({'datetime': time, 'y_true': y_true, 'y_pred': y_pred})
    final_predictions.to_sql('final_predictions', engine, if_exists='replace', index=False)
    typer.echo("Predictions saved to database.")



@app.command()
def plot_predictions(db_url: str = "sqlite:///pizza_sales.db"):
    engine = create_engine(db_url)

    final_predictions = pd.read_sql('SELECT * FROM final_predictions', engine)

    y_pred = final_predictions['y_pred']
    y_true = final_predictions['y_true']
    time = final_predictions['datetime']

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, y_pred, label='Prediction', color='darkorange')
    ax.plot(time, y_true, label='True', color='dodgerblue')
    ax.set_title('The Hawaiian sold quantities: True vs Predicted (30 days into the future)', fontsize=16)
    ax.set_ylabel('Quantity', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    plt.legend()
    plt.show()