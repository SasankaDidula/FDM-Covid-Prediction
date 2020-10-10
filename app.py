from flask import Flask, request, render_template, abort, Response
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

app = Flask(__name__)


def preprocessing(url):
    Df_dataset = pd.read_csv(url, error_bad_lines=False)

    Df_dataset.drop(['Country/Region', 'Lat', 'Long'], axis=1, inplace=True)

    col_num = 0
    TotalObjects = Df_dataset.shape[0]
    print("Column\t\t\t\t\t Null Values%")

    for x in Df_dataset:
        nullCount = Df_dataset[x].isnull().sum();
        nullPercent = nullCount * 100 / (TotalObjects)
        if nullCount > 0 and nullPercent > 30:
            col_num = col_num + 1
            Df_dataset.drop(x, axis=1, inplace=True)
            print(str(x) + "\t\t\t\t\t " + str(nullPercent))
    print("A total of " + str(col_num) + " deleted !")

    df1_transposed = Df_dataset.T  # or df1.transpose()
    df1_new = df1_transposed[2:266] - df1_transposed[2:266].shift()
    df1_new = df1_new.replace(np.nan, 0)
    return df1_new


def convert_date(df1_new, countryIndex, date):
    float_date = pd.to_datetime(date).toordinal()
    val = df1_new[countryIndex]
    date = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    Total_cases = val.values
    return date, Total_cases, [float_date]


def predict_cases(dates, cases, predictDate):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    cases = np.reshape(cases, (len(cases), 1))
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    svr_lin = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='linear', C=1e3, cache_size=7000))])
    svr_poly = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='poly', C=1e3, cache_size=7000, degree=2))])
    svr_rbf = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1e3, cache_size=7000, gamma=0.1))])

    # Fit regression model
    svr_lin.fit(dates, cases)
    svr_poly.fit(dates, cases)
    svr_rbf.fit(dates, cases)

    plt.scatter(dates, cases, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.title('Support Vector Regression')
    plt.legend()

    MeanErrorLin = 'Mean Squared Error Linear model: ' + str(mean_squared_error(cases, svr_lin.predict(dates)))
    MeanErrorRbf = 'Mean Squared Error RBF model: ' + str(mean_squared_error(cases, svr_rbf.predict(dates)))
    MeanErrorPoly = 'Mean Squared Error Polynomial model: ' + str(mean_squared_error(cases, svr_poly.predict(dates)))

    plt.savefig('static/' + 'image.png', dpi=600)
    plt.close()

    return svr_rbf.predict(predictDate)[0], svr_lin.predict(predictDate)[0], svr_poly.predict(predictDate)[
        0], MeanErrorLin, MeanErrorRbf, MeanErrorPoly


def predict_cases_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_cases = predict_cases(data[0], data[1], data[2])
    return predicted_cases


def predict_deaths(dates, deaths, predictDate):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    deaths = np.reshape(deaths, (len(deaths), 1))
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    regression_model = LinearRegression()
    # Fit linear_regression model
    regression_model.fit(dates, deaths)

    plt.scatter(dates, deaths, c='k', label='Data')
    plt.plot(dates, regression_model.predict(dates), c='g', label='Linear model')
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.title('Linear Regression')
    plt.legend()

    MeanErrorLin = 'Mean Error Linear model: ' + str(mean_squared_error(deaths, regression_model.predict(dates)))

    plt.savefig('static/' + 'image.png', dpi=600)
    plt.close()

    return regression_model.predict(predictDate)[0], MeanErrorLin


def predict_deaths_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_deaths = predict_deaths(data[0], data[1], data[2])
    return predicted_deaths


def predict_recovered(dates, recovered, x):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    x = np.reshape(x, (len(x), 1))

    # Fitting Polynomial Regression to the dataset
    polynominal_regg = PolynomialFeatures(degree=3)
    x_Polynom = polynominal_regg.fit_transform(dates)

    leniar_regg = LinearRegression()
    leniar_regg.fit(x_Polynom, recovered)
    # Visualizing the Polymonial Regression results
    plt.scatter(dates, recovered, c='k', label='Data')
    plt.plot(dates, leniar_regg.predict(x_Polynom), c='b', label='RBF model')

    MeanErrorLin = 'Mean Error Polynomial Regression model: ' + str(mean_squared_error(recovered, leniar_regg.predict(x_Polynom)))

    plt.savefig('static/' + 'image.png', dpi=600)
    plt.close()

    return leniar_regg.predict(polynominal_regg.fit_transform(x)), MeanErrorLin


def predict_recovered_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_recovered = predict_recovered(data[0], data[1], data[2])
    return predicted_recovered

@app.route('/predict', methods=['POST'])
def predict():
    features = [formValues for formValues in request.form.values()]
    kwargs = {}
    Output = []
    if int(features[1]) == 0:
        Output = predict_cases_country(int(features[0]), features[2])
        kwargs = {'div': 'RBF model:' + str(Output[0]) + ', Linear model:' + str(
            Output[1]) + ', Polynomial model:' + str(
            Output[2]),
                  'MeanError': str(Output[3]) + ', ' + str(
                      Output[4]) + ', ' + str(Output[5])}

    elif int(features[1]) == 2:
        Output = predict_deaths_country(int(features[0]), features[2])
        kwargs = {'div': 'Linear Regression model:' + str(Output[0]),
                  'MeanError': str(Output[1])}

    else:
        Output = predict_recovered_country(int(features[0]), features[2])
        kwargs = {'div': 'Polynomial Regression model:' + str(Output[0]),
                  'MeanError': str(Output[1])}

    return render_template('main.html', **kwargs)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
