from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    prediction = None  # Initialize prediction variable
    if request.method == 'POST':
        # Extract user input from the form
        RH = request.form.get('RH', type=float)
        Temperature = request.form.get('Temperature', type=float)
        Rain = request.form.get('Rain', type=float)
        WindSpeed = request.form.get('WindSpeed', type=float)
        FFMC = request.form.get('FFMC', type=float)
        DMC = request.form.get('DMC', type=float)
        ISI = request.form.get('ISI', type=float)
        Classes = request.form.get('Classes', type=int)
        Region = request.form.get('Region', type=int)
        DC = request.form.get('DC', type=float)  # Add DC input here

        # Create an instance of CustomData with user input
        data = CustomData(RH, Temperature, Rain, WindSpeed, FFMC, DMC, ISI, Classes, Region, DC)

        # Convert the input data to DataFrame
        final_new_data = data.get_data_as_dataframe()

        # Create a prediction pipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        # Round the prediction result for readability
        prediction = round(pred[0], 2)  # Store the prediction to display it later

    # Render the template with the prediction (if any)
    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
