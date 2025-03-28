import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import fastapi
from transformers.data.data_collator import tolist

app = fastapi.FastAPI()


@app.get("/predict")
def predict():
    df = pd.read_csv(
        'newhousing.csv')

    #df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

    reg = LinearRegression()

    # dropping the irrelevant columns in the multivariate dataset
    reg.fit(df.drop(['price', 'hotwaterheating',
                     'airconditioning', 'prefarea', 'mainroad', 'semi-furnished', 'unfurnished'], axis='columns'), df.price)

    # predicting with area, bedroom, bathrooms, 'basement', stories, guestroom, parking, areaperbedroom, bbratio
    print(reg.predict(
        [[5500, 3, 2, 2, 1, 0, 1, 1833.22, 0.667]]))

    # Saving model to disk
    pickle.dump(reg, open('model.pkl', 'wb'))

    # Loading model to compare the results
    model = pickle.load(open('model.pkl', 'rb'))

    return {"prediction": model.predict([[5500, 3, 2, 2, 1, 0, 1, 1833.22, 0.667]]).tolist()}
