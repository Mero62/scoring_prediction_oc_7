from flask import Flask, jsonify, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import shap

app = Flask(__name__)

# Load the model (LightGBM) and data
model_lightgbm = pickle.load(open('model_lightGBM.pkl', 'rb'))
data = pd.read_csv('test_df_sample_500.csv')
data_train = pd.read_csv('train_df_sample_5000.csv')

cols = data.select_dtypes(['float64']).columns
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])
cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])

explainer = shap.TreeExplainer(model_lightgbm['classifier'])


@app.route('/')
def welcome():
    return 'Welcome to the scoring prediction API'


@app.route('/<int:client_id>')
def check_client_id(client_id):
    if client_id in list(data['SK_ID_CURR']):
        return jsonify(True)
    else:
        return jsonify(False)


@app.route('/prediction/<int:client_id>')
def get_prediction(client_id):
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model_lightgbm.predict_proba(info_client)[0][1]
    return jsonify(prediction)


@app.route('/similar_clients/<int:client_id>')
def get_data_voisins(client_id):
    features = list(data_train_scaled.columns)
    features.remove('SK_ID_CURR')
    features.remove('TARGET')

    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn.fit(data_train_scaled[features])
    reference_id = client_id
    reference_observation = data_scaled[data_scaled['SK_ID_CURR'] == reference_id][features].values
    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = data_train.iloc[indices[0], :]

    return df_voisins.to_json()


@app.route('/shaplocal/<int:client_id>')
def shap_values_local(client_id):
    client_data = data_scaled[data_scaled['SK_ID_CURR'] == client_id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    shap_val = explainer(client_data)[0][:, 1]

    return jsonify({
        'shap_values': shap_val.tolist(),
        'base_value': shap_val.base_values,
        'data': client_data.values.tolist(),
        'feature_names': client_data.columns.tolist()
    })


@app.route('/shap/')
def shap_values():
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return jsonify({
        'shap_values_0': shap_val[0].tolist(),
        'shap_values_1': shap_val[1].tolist()
    })


if __name__ == '__main__':
    app.run()
