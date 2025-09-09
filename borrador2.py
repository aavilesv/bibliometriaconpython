import pandas as pd

data = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")
#data['institution'] = data['country']
data['institutionWithCountry'] = data['country']

data.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv" ,index=False)
