import requests
import numpy as np
import pandas as pd


if __name__ == '__main__':
	# intermediate file with the computed features from the pointcloud file 
	features_pcfile = "/Users/arnab/devwork/lgcwork/basicDNN/input/int_pcfile.csv"
	df = pd.read_csv(features_pcfile)

	# convert the dataframe to string
	payload = df.to_json(orient='values')
	payload = "{\"input\": " + payload + "}"

	headers = {
	    'accept': "application/json",
	    'content-type': "application/json"
	    }

	# make a 'POST' request to the prediction server running on localhost:8080
	url = "http://localhost:8080/predict/basicdnn/invoke"
	response = requests.request("POST", url, data=payload, headers=headers)

	print(response.text)