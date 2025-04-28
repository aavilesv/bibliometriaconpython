import pandas as pd
import re

# Cargar el archivo CSV
#df = pd.read_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv")

df = pd.read_csv("G:\\Mi unidad\\2025\\master kevin castillo\\artículo nuevo\\data\\datawos_scopuslematizar.csv")

# Diccionario de palabras clave a reemplazar: clave = palabra a buscar (en minúsculas), valor = palabra de reemplazo
palabras_clave_reemplazo = {
    "internet of thing": "internet of things",
    "learn system": "learning system",
    "learn algorithm": "learning algorithm",

    "decision make": "decision making",
    "predictive analytic": "predictive analytics",
    "big datum": "big data",
    "datum acquisition": "data acquisition",
    "datum set": "dataset",
    "satellite datum": "satellite data",
    "datum mining": "data mining",
    
    "ml algorithm":"machine learning",
    "datum handle": "data handling",
    "datum fusion": "data fusion",
    "datum analytic": "data analytics",
    "datum integration": "data integration",
    "normalize difference vegetation index": "normalized difference vegetation index",
    "ndvi": "normalized difference vegetation index",
    "cnn": "convolutional neural network",
    "deep cnn": "convolutional neural network",
    "deep leanring":"deep learning",
    "deep neural network":"deep learning",
    "neural network":"artificial neural network",
    "deep":"deep learning",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "gru": "gated recurrent unit",
    "ml": "machine learning",
    "iot": "internet of things",
   "uav": "unmanned aerial vehicle",
    "uas": "unmanned aerial system",
    "machine learn": "machine learning",
  "machine learn algorithm": "machine learning",
  "machine learn technique": "machine learning",
  "machine learn classifier": "machine learning",
  "machine learn prediction": "machine learning",
    "learn machine": "machine learning",

  "auto machine learning": "automated machine learning",
  "automate machine learning": "automated machine learning",
  "extreme learn machine": "extreme learning machine",
    "improve extreme learning machine": "extreme learning machine",
  "ensemble learning model": "ensemble learning",
    "hybrid machine learning model": "hybrid machine learning",

  "ensemble machine learning algorithm": "ensemble learning algorithm",
  "extreme learn machine": "extreme learning machine",
  "support vector machine model":"support vector machine",
  "support vector machine regression" :"support vector machine",

  "federate learn": "federated learning",
  "few shoot learn": "few-shot learning",
  "transfer learn": "transfer learning",
  "transfer learn model": "transfer learning model",
  "deep learn": "deep learning",
  "deep learn change detection": "deep learning change detection",
  "deep learn multi layer perceptron": "deep learning multi-layer perceptron",
  "deep learn neural network": "deep learning neural network",
  "deep learn predictor": "deep learning predictor",
  "deep learn vision raspberry pi4": "deep learning vision",
  "semi supervise learning": "semi-supervised learning",
  "self supervise learning": "self-supervised learning",
  "supervise learning": "supervised learning",
  "unsupervise learning": "unsupervised learning",
  "e learn": "e-learning",
  "learn": "learning",
  "learn system": "learning system",
  "learn algorithm": "learning algorithm",
  "learn model": "learning model",
  "learn technique": "learning technique",
  "learn classifier": "learning classifier",
    "remote sense datum": "remote sensing data",
    "image process": "image processing",
    "image preprocesse": "image preprocessing",
    "real time": "real-time",
    "real time system": "real-time system",
    "real time detection": "real-time detection",
    "mean square error": "mean squared error",
    "root mean square error": "root mean squared error",
       
   "satellite datum": "satellite data",
    "datum acquisition": "data acquisition",
    "datum set": "dataset",
    "datum mining": "data mining",
    "datum handle": "data handling",
    "datum fusion": "data fusion",
    "datum analytic": "data analytics",
    "datum integration": "data integration",
    "real world agricultural datum": "real world agricultural data",
    "imagery datum": "imagery data",
    "measurement datum": "measurement data",
    "historical datum": "historical data",
    "sensor datum": "sensor data",
    "satellite sensor datum": "satellite sensor data",
    "satellite remote sense datum": "satellite remote sensing data",
    "remote sense datum": "remote sensing data",
    "convolutional network":"convolutional neural network",
    "advanced convolutional neural network":"convolutional neural network",
    "advanced cnn":"convolutional neural network",
    "internet of thing": "internet of things",
    "iot": "internet of things",
    "thing iot": "internet of things",
    #"uav": "unmanned aerial vehicle",
    
    "remote sense technique": "remote sensing technology",

        "hyperspectral image": "hyperspectral imagery",
        
 "hyperspectral imagery": "hyperspectral imagery",
 "gf 5 hyperspectral image": "gf 5 hyperspectral imagery",
 "hyperspectral reflectance image": "hyperspectral reflectance",
 "hyperspectral imaging datum": "hyperspectral imaging",
 "hyperspectral remote sense image": "hyperspectral remote sensing",
 "hyperspectral remote sense technology": "hyperspectral remote sensing",
 "hyperspectral remote sense datum": "hyperspectral remote sensing",
 "hyperspectral datum": "hyperspectral data",
 
     "reinforcement learning":"deep reinforcement learning",
     
    "remote sense technology": "remote sensing technology",
    "uas": "unmanned aerial system",
    "ndvi": "normalized difference vegetation index",
    "normalize difference vegetation index": "normalized difference vegetation index",
    "cnn": "convolutional neural network",
    "gi": "geographic information",
    "satellite datum": "satellite data",
    "datum acquisition": "data acquisition",
    "datum mining": "data mining",
    "datum fusion": "data fusion",
    "big datum": "big data",
    "datum set": "dataset",
    "datum handle": "data handling",
    "datum analysis": "data analysis",
    "mean square error": "mean squared error",
    "root mean square error": "root mean squared error",
    "predictive analytic": "predictive analytics",
    "learn system": "learning system",
    "learn algorithm": "learning algorithm",
    "decision make": "decision making",
    "transfer learn": "transfer learning",
    "e learn": "e-learning",
    "real time": "real-time",
    "real time system": "real-time system",
    "real time monitor": "real-time monitoring",
    
    
    "convolution": "convolutional",
    "convolution neural network": "convolutional neural network",

      "precision agricultura": "precision agriculture",
      
    "agricolus": "agricultural",
    "efficient agricultura": "efficient agriculture",
    "precision agricultura": "precision agriculture",
       
    "smart precision agriculture": "precision agriculture",
    "smart agriculture system": "smart agriculture",
    "smart agriculture management": "smart agriculture",

  "agriculture sector": "agricultural sector",
    "agriculture system": "agricultural system",
    "agriculture management": "agricultural management",
    "agriculture technology": "agricultural technology",
    "agriculture application": "agricultural application",
    "agriculture monitoring": "agricultural monitoring",
    "agriculture robot": "agricultural robot",
    "agriculture crop": "agricultural crop",
    "agriculture decision": "agricultural decision",
    "agriculture production": "agricultural production",

    # “agriculture x.0” versions
    "agriculture 4": "agriculture 4.0",
    "agriculture 40": "agriculture 4.0",
    "agriculture 50": "agriculture 5.0",
    "agriculture 20": "agriculture 2.0",

    # iot en agricultura
    "agricultural internet of thing":      "agricultural internet of things",
    "agriculture internet of thing":       "agricultural internet of things",
    "internet of agricultural thing":      "agricultural internet of things",
    "iot in agriculture":                  "agricultural internet of things",
    "agricultural iot":                    "agricultural internet of things",
    "cluster base agricultural iot":       "agricultural internet of things",
    "iot base precision agriculture":      "agricultural internet of things",
    "precision agriculture iot sensor":    "agricultural internet of things",

    # e-agriculture
    "e agriculture": "e-agriculture",
    "agriculture automation": "automation",
    "uavdrone": "drone",
   "autonomous farming drone": "drone",
   "farm drone": "drone",
   "drone and agriculture": "drone",
   "drone base imaging": "drone imaging",
   "drone image": "drone imaging",
   "drone sensing": "drone remote sensing",
    # british → us
 "yield modelling": "yield modeling",
 # mapeo de “model” vs “modeling”
 "yield model": "yield modeling",

 # forecasting
 "yield forecast": "yield forecasting",
 "crop yield forecasting": "yield forecasting",
 "crop yield forecast": "yield forecasting",

 # prediction
 "yield prediction model": "yield prediction",
 "boost crop yield prediction": "yield prediction",
 "empirical yield prediction": "yield prediction",

 # estimation
 "early yield estimation": "yield estimation",
 "grain yield estimation": "yield estimation",
 "crop yield estimation": "yield estimation",
 "yield estimate": "yield estimation",

 # map/mapping
 "yield mapping": "yield map",

 # monitoring
 "yield monitor": "yield monitoring",
 "yield monitor system": "yield monitoring",
 "yield monitoring system": "yield monitoring",

 # valores genéricos de “yield”
 "average yield": "yield",
 "dry matter yield": "yield",
 "grain yield": "yield",
 "biennial yield": "yield",
 "high quality yield": "yield",
 "low yield": "yield",

 # corrección de typo
 "yield and cost benefit analyse": "yield and cost benefit analysis",

 # variantes de alfalfa
 "alfalpha yield and quality": "alfalfa yield and quality",
 "alfalfa yield and quality": "alfalfa yield and quality",
  # variantes de “yield prediction”
   "crop yield prediction": "yield prediction",
   "yield prediction model": "yield prediction",
   "boost crop yield prediction": "yield prediction",
   "empirical yield prediction": "yield prediction",
   "maize yield prediction": "yield prediction",
   "soybean yield prediction": "yield prediction",
   "corn yield prediction": "yield prediction",
   "cotton yield prediction": "yield prediction",
   "rice yield prediction": "yield prediction",
   "sugarcane yield prediction": "yield prediction",
   "paddy yield prediction": "yield prediction",
   "peanut yield prediction": "yield prediction",

   # agrupación de “prediction accuracy”
   "prediction performance":      "prediction accuracy",
   "prediction quality":          "prediction accuracy",
   "crop prediction accuracy":    "prediction accuracy",

   # ortografía us de “modelling”
   "prediction modelling": "prediction modeling",

   # sistemas de predicción
   "crop prediction system": "prediction system",

   # predicción de producción
   "crop production prediction": "production prediction",
   "predictive model":"predictive modeling",
   "predictive analytic":"predictive analytics",
   # smart farming variantes
   "smart farm": "smart farming",
   "smart farming system": "smart farming",
   "smart farming decision": "smart farming",
   "smart farming industry": "smart farming",
   "trend in smart farming": "smart farming",

   # smallholder farming variantes
   "smallholder farm": "smallholder farming",

   # sustainable farming variantes
   "sustainable farming practice": "sustainable farming",

   # farm management
   "farm management system": "farm management",

   # experimentación en la finca
   "on farm experiment": "on farm experimentation",

   # robótica agrícola
   "farming bot": "farmbot",

   # agricultura mixta
   "mix farming": "mixed farming",
   "digital farming": "digital agriculture",
  "digital image": "digital imaging",
  "digital imagery": "digital imaging",
  "digital greenhouse system": "digital greenhouse",
  
  
"unmanned vehicle":"unmanned aerial vehicle", 
"autonomous aerial vehicle": "unmanned aerial vehicle",
"autonomous unmanned aerial vehicle":"unmanned aerial vehicle",
"unmanned arial vehicle":"unmanned aerial vehicle",
    "small unmanned aerial vehicle": "unmanned aerial vehicle",
    "aerial vehicle": "unmanned aerial vehicle",
    "multiple linear regression method": "multiple linear regression",
  "multiple linear regression model": "multiple linear regression",
  "linear regression model": "linear regression",
  "linear regression analysis": "linear regression",
  "near neighbor search": "k-nearest neighbor",
  "near neighbor analysis": "k-nearest neighbor",
  "k near neighbor": "k-nearest neighbor",
  "knn": "k-nearest neighbor",
  "svm":"support vector machine",
  "svm classifier":"support vector machine",
  "svnn":"support vector machine",
  "support vector machine model":"support vector machine",
  "support vector machine regression":"support vector machine",


}

def reemplazar_palabras_clave(column, diccionario_reemplazo):
    """
    Recorre cada celda de la columna, separa los términos (suponiendo que estén separados por ';'),
    y reemplaza aquellos que coincidan (ignorando mayúsculas/minúsculas) por el valor correspondiente del diccionario.
    """
    def process_cell(cell):
        # Si la celda es una cadena, separamos usando el delimitador ';'
        if isinstance(cell, str):
            terminos = [termino.strip() for termino in cell.split(';') if termino.strip()]
        # Si ya es una lista, la usamos directamente
        elif isinstance(cell, list):
            terminos = [str(termino).strip() for termino in cell if str(termino).strip()]
        else:
            terminos = []
        
        terminos_modificados = []
        for termino in terminos:
            # Convertimos el término a minúsculas para la comparación
            termino_lower = termino.lower()
            if termino_lower in diccionario_reemplazo:
                # Reemplazamos por el valor definido en el diccionario
                terminos_modificados.append(diccionario_reemplazo[termino_lower])
            else:
                terminos_modificados.append(termino)
        return '; '.join(terminos_modificados)
    
    return column.apply(process_cell)

# Aplicar la función a las columnas "Index Keywords" y "Author Keywords"
df['Index Keywords'] = reemplazar_palabras_clave(df['Index Keywords'], palabras_clave_reemplazo)
df['Author Keywords'] = reemplazar_palabras_clave(df['Author Keywords'], palabras_clave_reemplazo)
#df['bothKeywords'] =  reemplazar_palabras_clave(df['bothKeywords'], palabras_clave_reemplazo)
# --- A PARTIR DE AQUÍ, EL CÓDIGO NUEVO PARA REEMPLAZOS PARCIALES ---

def reemplazar_parciales(column, patrones):
    """
    Recorre cada celda de la columna, y por cada patrón (regex) en 'patrones',
    realiza la sustitución indicada.
    - 'patrones' debe ser una lista de tuplas (pattern, replacement).
    - Se ignoran mayúsculas/minúsculas (flags=re.IGNORECASE).
    """
    def process_cell(cell):
        if isinstance(cell, str):
            # Aplica todos los patrones de reemplazo parcial
            for patron, nuevo_texto in patrones:
                cell = re.sub(patron, nuevo_texto, cell, flags=re.IGNORECASE)
        return cell

    return column.apply(process_cell)

# Ejemplo de un arreglo de reemplazos parciales
# Cada tupla es (expresión_regular, texto_reemplazo)
# Aquí solo se incluye 'datum' -> 'data', pero puedes añadir más.
patrones_parciales = [
    (r'datum', 'data'),  # Reemplaza 'datum' donde aparezca (ignora mayúsculas)
    # Si necesitas más reemplazos parciales:
    # (r'algunaSubcadena', 'otroTexto'),
    # (r'pattern', 'replacement'),
    # ...
]
# 2) Después, los reemplazos parciales:
#df['bothKeywords'] = reemplazar_parciales(df['bothKeywords'], patrones_parciales)
df['Index Keywords'] = reemplazar_parciales(df['Index Keywords'], patrones_parciales)
df['Author Keywords'] = reemplazar_parciales(df['Author Keywords'], patrones_parciales)
# Guardar el DataFrame modificado en un nuevo archivo CSV
#df.to_csv("G:\\Mi unidad\\2024\\SCientoPy\\ScientoPy\\dataPre\\papersPreprocessed.csv", index=False)
df.to_csv("G:\\Mi unidad\\2025\\master kevin castillo\\artículo nuevo\\data\\wos_scopus_reemplazado.csv", index=False)
print("Palabras clave reemplazadas y nuevo archivo guardado.")