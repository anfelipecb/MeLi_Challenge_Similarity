
from pathlib import Path  #Librería para manejar las rutas de los archivos

data_dir= Path('/Users/anfelipecb/GitHub/MeLi/MeLi_Challenge_Similarity/data')
data_final= data_dir / 'final' #Final data
data_final_imgs= data_dir / 'final/imgs' #Final data
data_processed=data_dir / 'processed'   #Intermediate data
data_raw=data_dir / 'raw' #Original data 
#print(data_final.resolve())

# Configuración de la API
API_URL="https://api.mercadolibre.com"

