from pycaret.regression import *
import pickle
import pandas as pd
bikerentals = pd.read_csv('https://aka.ms/bike-rentals')

job2 = load_experiment('Bike rentals experiment', data = bikerentals)
modelo = open('modelo_bike_rentals_pycaret','rb')

modelo_final = pickle.load(modelo)

modelo.close()
create_app(modelo_final)