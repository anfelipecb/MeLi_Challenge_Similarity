import requests
from .config import API_URL
import pandas as pd
from pandas import json_normalize

def get_data_API_items(category_id, limit=None, offset=0):
    """
    Obtiene los productos de la categoría indicada a partir de la API de MercadoLibre API para una categoría específica y hasta un limite determinado 
    (default= Ninguno). Si se corre sin limete está botando igual solo 1000 porque la API tiene restricciones. 
    
    Parametros:
    - category_id: El ID de la categoría de la que se sacan los productos.
    - api_url: La URL base de la API MELI.
    - limit: Numero maximo de productos para reunir.
    
    Devuelve (return):
    - Data frame de pandas conteniendo todos los productos hasta el limite especificado
    """
    #offset = 0 # el máximo permitido de la API pública es 1000 pero esto funciona para todos. 
    products_list = []  #Lista vaciía de productos 
    
    # Hacer una solicitud inicial para determinar el número total de productos si no se especifica un límite
    if limit is None:
        initial_url = f'{API_URL}/sites/MLA/search?category={category_id}&offset={offset}'
        initial_response = requests.get(initial_url)
        if initial_response.status_code == 200:
            total_available = initial_response.json().get('paging', {}).get('total', 0)
            limit = total_available
            print(f'El limite de items es {limit}')
        else:
            print(f"Error obteniendo datos iniciales: HTTP {initial_response.status_code}")
            return pd.DataFrame()
    
    while offset < limit:
        url = f'{API_URL}/sites/MLA/search?category={category_id}&offset={offset}'
        response = requests.get(url)
        
        #Verificamos la respuesta exitosa 
        if response.status_code !=200: 
            print(f"Error obteniendo los datos: {response.status_code}")
            break
        
        data = response.json()
        
        #Verificamos la respuesta exitosa tiene results
        if 'results' not in data: 
            print('No hay resultados encontrados en la respuesta')
            break
        
        # Revisa que no hay mas productos para traer
        if not data['results']:
            print("No hay más productos para mostrar")
            break
        
        products_list.extend(data['results'])
        
        # actualización del offset
        offset += len(data['results']) #iteracion de 50 en 50 en la práctica, pero más flexible, por si la última página no tiene los 50 sino menos
        size=len(data['results'])
        print(f'va en el ofsett: {offset} con tamaño:{size}')
    #Normalizamos 
    df = pd.json_normalize(products_list)
    
    # Los atributos que me parecen importantes 
        # Aquí: df.apply aplicara a cada fila la funcion lambda para transformar la estructura de datos anidada a la base que estamos armando
    for attr_name in ['BRAND', 'LINE', 'MODEL', 'PACKAGE_LENGTH', 'PACKAGE_WEIGHT']:
        df[attr_name.lower()] = df.apply(
            lambda row: next(
                (attr['value_name'] for attr in row['attributes'] if attr['id'] == attr_name), 
                None
            ), 
            axis=1
        )
    # Quitamos atributos que no voy a usar para hacer menos espacio
    df.drop(columns=['attributes'], inplace=True)
    
    return df #La base de datos como la queremos