
from pathlib import Path  #Librería para manejar las rutas de los archivos

data_dir= Path('../../data')
data_cleaned= data_dir / 'cleaned' #Final data
data_processed=data_dir / 'processed'   #Intermediate data
data_raw=data_dir / 'raw' #Original data 

# Configuración de la API
API_URL="https://api.mercadolibre.com"

