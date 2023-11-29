import joblib
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

model = joblib.load('model.pickle')


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


scaler = joblib.load('scaler.pickle')


def transform_input_data(item):
    if isinstance(item, pd.Series):
        item_dict = item
    else:
        item_dict = item.dict()

    df = pd.DataFrame([item_dict])

    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').replace('', np.nan).astype(float)

    df['torque'] = df['torque'].str.lower()
    df['torque'] = df['torque'].str.replace(r'\s|kgm|\)', '', regex=True)
    df['torque'] = df['torque'].str.replace(r'nm|n\(|at|@', 'n', regex=True)
    df['torque'] = df['torque'].str.replace(r'~|/', '-', regex=True)
    df['torque'] = df['torque'].str.replace('+/-', '-', regex=False)
    df['torque'] = df['torque'].astype(str)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(
        lambda x: pd.Series(x.split('n', maxsplit=1)) if 'n' in x else (
            pd.Series(x.split('-')) if '-' in x else pd.Series([x, np.nan])))
    df['torque'] = df['torque'].str.replace('kgf', '')
    df['torque'] = df['torque'].replace('', np.nan)
    df['torque'] = df['torque'].astype(float)
    df['torque'] = df['torque'].apply(lambda x: x * 9.8 if x < 40 else x)
    df['max_torque_rpm'] = df['max_torque_rpm'].str.replace('[a-zA-Z,\\(+]', '', regex=True)
    df['max_torque_rpm'] = df['max_torque_rpm'].astype(str)
    df['max_torque_rpm'] = df['max_torque_rpm'].apply(
        lambda x: sum(map(float, x.split('--'))) if x.count('-') == 2 and x != '' else (
            max(map(float, x.split('-'))) if x.count('-') == 1 and x != '' else (
                float(x) if x.count('-') == 0 and x != '' else x)))
    df['max_torque_rpm'] = df['max_torque_rpm'].apply(lambda x: float(x) if x else np.nan)

    df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
    df = df.drop(['name', 'selling_price'], axis=1)

    all_fuels = ['CNG', 'Diesel', 'LPG', 'Petrol']
    all_seller_types = ['Dealer', 'Individual', 'Trustmark Dealer']
    all_transmissions = ['Automatic', 'Manual']
    all_owners = ['First Owner', 'Fourth & Above Owner', 'Second Owner', 'Test Drive Car', 'Third Owner']
    all_seats = [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 14.0]
    all_brands = ['Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda',
                  'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti',
                  'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota',
                  'Volkswagen', 'Volvo']

    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand'],
                        drop_first=False)

    all_categories = ['fuel_' + fuel for fuel in all_fuels] + \
                     ['seller_type_' + seller_type for seller_type in all_seller_types] + \
                     ['transmission_' + transmission for transmission in all_transmissions] + \
                     ['owner_' + owner for owner in all_owners] + \
                     ['seats_' + str(int(seat)) for seat in all_seats] + \
                     ['brand_' + brand for brand in all_brands]

    for category in all_categories:
        if category not in df.columns:
            df[category] = 0

    other_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    all_columns = other_columns + all_categories

    df = df.reindex(columns=all_columns)

    df['horsepower_per_litre'] = df['max_power'] / df['engine']
    df['year_squared'] = df['year'] ** 2
    df['mileage_squared'] = df['mileage'] ** 2

    for fuel_type in ['fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'fuel_CNG']:
        df[f'{fuel_type}_max_power'] = df[fuel_type] * df['max_power']

    df_scaled = scaler.transform(df)

    return pd.DataFrame(df_scaled)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_transformed = transform_input_data(item)
    log_prediction = model.predict(item_transformed)
    prediction = np.exp(log_prediction)

    return round(prediction[0], 1)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_transformed = [transform_input_data(item) for item in items]
    items_df = pd.concat(items_transformed, ignore_index=True)
    log_predictions = model.predict(items_df)
    predictions = np.exp(log_predictions)
    rounded_predictions = [round(prediction, 1) for prediction in predictions]

    return rounded_predictions


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    df_transformed = pd.concat([transform_input_data(row) for _, row in df.iterrows()])

    log_predictions = model.predict(df_transformed)
    predictions = np.exp(log_predictions)
    rounded_predictions = [round(prediction, 1) for prediction in predictions]

    df['predictions'] = rounded_predictions
    output_filename = "predictions.csv"
    df.to_csv(output_filename, index=False)

    return FileResponse(output_filename, filename=output_filename)
