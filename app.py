from flask import Flask, request, jsonify
import joblib # বা আপনার মডেল অনুযায়ী pickle/tf

app = Flask(__name__)
model = joblib.load('water_quality_model.pkl') # আপনার মডেল ফাইল

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # ESP32 থেকে আসা JSON ডেটা
    prediction = model.predict([data['features']])
    return jsonify({'feedback': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
