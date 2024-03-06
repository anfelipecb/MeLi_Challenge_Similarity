import requests
from .config import API_URL, data_final, data_processed, data_raw
import pandas as pd
from pandas import json_normalize
import time 
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity


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

#Embeddings de imagenes 
#Ajustamos las funciones para que se adapten a diferentes modelos que tienen diferentes funciones de preprocesamiento. 

def download_and_preprocess_image_for_model(image_url, preprocess_input_func, target_size):
    """ Función genérica para descargar una imagen desde una URL, redimensionarla a un target size, y aplicar el preprocesamiento
        que indica cada modelo
        
        Parametros:
        - image_url: La url de la imagen 
        - preprocess_input_func: funcion de preprocesamiento de TensorFlow
        - target_size: Dimesiones a las que se pasa la imagen 
        
        Devuelve (return):
        - la imagen preprocessada 
    """
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) #agregamos dimension para convertir el array en batch de tamaño 1
        image = preprocess_input_func(image)
        return image
    except Exception as e:
        print(f"Error al descargar o procesar la imagen {image_url}: {e}")
        return None


#La funcion que usa la anterior y extrae los embeddings dependiendo del modelo 
def extract_features_for_model(df, model, preprocess_input_func, target_size):
    """ Extrae características de una lista de imágenes usando un modelo preentrenado.
    
        Parametros: 
        - df= dataframe donde están las urls de las imagenes en la variable thumbnail
        - model: la red neuronal pre entrenada 
        - preprocess_input_func: la funcion específica de preprocesamiento de los inputs
        - target_size: el tamaño target de las imagenes 
    """
    
    start_time = time.time()
    
    features_list = []
    ids = []
    for _, row in df.iterrows():
        preprocessed_image = download_and_preprocess_image_for_model(row['thumbnail'], preprocess_input_func, target_size)
        if preprocessed_image is not None:
            features = model.predict(preprocessed_image)
            features_flattened = features.flatten()
            features_list.append(features_flattened)
            ids.append(row['id'])
    
    end_time = time.time()
    print(f"Tiempo total para procesar {len(df)} imágenes con {model.name}: {end_time - start_time} segundos")
    
    return ids, features_list


#Visualizar los embeddings 

def visualizar_embeddings(features_list,out_name, dims=2, method='PCA'):
    """
    Función para visualizar embeddings en 2 o 3 dimensiones usando PCA o t-SNE.

    Parámetros:
    - features_list: Lista o array de Numpy con los embeddings.
    - dims: Dimensiones para la visualización (2 o 3).
    - method: Método de reducción de dimensionalidad ('PCA' o 't-SNE').
    """
    
    # Verificar que dims sea 2 o 3
    if dims not in [2, 3]:
        raise ValueError("dims debe ser 2 o 3.")
    
    # Reducción de dimensionalidad
    if method == 'PCA':
        model = PCA(n_components=dims)
    elif method == 't-SNE':
        model = TSNE(n_components=dims, learning_rate='auto', init='random')
    else:
        raise ValueError("method debe ser 'PCA' o 't-SNE'.")
    features_list_np=np.array(features_list)
    embeddings_reducidos = model.fit_transform(features_list_np)
    
    # Gráfico
    fig = plt.figure(figsize=(8, 8))
    if dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_reducidos[:, 0], embeddings_reducidos[:, 1], embeddings_reducidos[:, 2])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
    else:
        plt.scatter(embeddings_reducidos[:, 0], embeddings_reducidos[:, 1])
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
    
    plt.title(f'Visualización de Embeddings con {method}')
    
    #Guardamos
    plt.savefig(f'{data_processed}/{out_name}.png')
    plt.show()
    
def Guardar_arrays(feature_list,out_name): 
    """Guarda los arrays para usarlos despues en otros notebooks 

    Args:
        feature_list (_type_): la lista de embeddings
        out_name (_type_): El nombre que se le pondrá al archivo
    """
    np.save(f'{data_final}/{out_name}.npy', feature_list)
    
# Reducir dimensionalidad de clusters 
def ajustar_clustering_y_reducir(features, n_components, modelo_dim_reduc, n_clusters):
    """
    Reduce la dimensionalidad de los features y ajusta KMeans en las características reducidas.

    Parámetros:
    - features: array-like, forma (n_samples, n_features)
        Las características combinadas a ser reducidas y clusterizadas.
    - n_components: int
        El número de componentes a los que reducir las características.
    - modelo_dim_reduc: {'PCA', 't-SNE'}
        El modelo de reducción de dimensionalidad a utilizar.
    - n_clusters: int
        El número de clusters para KMeans.
    
    Retorna:
    - modelo_kmeans: objeto KMeans ajustado.
    - features_reducidas: array-like, forma (n_samples, n_components)
        Las características reducidas.
    """
    # Reducción de dimensionalidad
    if modelo_dim_reduc == 'PCA':
        reducer = PCA(n_components=n_components)
    elif modelo_dim_reduc == 't-SNE':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("modelo_dim_reduc debe ser 'PCA' o 't-SNE'")
    
    features_reducidas = reducer.fit_transform(features)
    
    # Ajustar KMeans
    modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_reducidas)
    
    return modelo_kmeans, features_reducidas



#Metricas de calidad de clusters 


def analizar_clusters(modelo_kmeans, features_reducidas):
    """
    Calcula el score de silueta promedio y las estadísticas detalladas por cluster
    para un modelo KMeans dado y características reducidas.

    Parámetros:
    - modelo_kmeans: objeto KMeans ajustado.
    - features_reducidas: array-like, las características reducidas sobre las cuales se ajustó KMeans.
    
    Retorna:
    - Un DataFrame con las estadísticas agregadas por cluster.
    """
    cluster_labels = modelo_kmeans.labels_
    silhouette_avg = silhouette_score(features_reducidas, cluster_labels)
    print(f"El score de silueta promedio es: {silhouette_avg}")

    silhouette_vals = silhouette_samples(features_reducidas, cluster_labels)
    df_clusters = pd.DataFrame({'cluster': cluster_labels, 'silhouette': silhouette_vals})
    
    for i in np.unique(cluster_labels):
        cluster_features = features_reducidas[cluster_labels == i]
        df_clusters.loc[df_clusters['cluster'] == i, 'features_mean'] = np.mean(cluster_features, axis=1)
        df_clusters.loc[df_clusters['cluster'] == i, 'features_std'] = np.std(cluster_features, axis=1)
    
    cluster_stats = df_clusters.groupby('cluster').agg(
        silhouette_mean=('silhouette', 'mean'),
        silhouette_std=('silhouette', 'std'),
        features_mean=('features_mean', 'mean'),
        features_std=('features_std', 'mean')
    ).reset_index()

    return cluster_stats


def calcular_similitud_coseno_clusters(modelo_kmeans, features_reducidas):
    """
    Calcula la similitud media del coseno dentro de cada cluster para un modelo KMeans dado.

    Parámetros:
    - modelo_kmeans: Modelo KMeans ajustado.
    - features_reducidas: Características reducidas sobre las que se ajustó el modelo KMeans.
    
    Retorna:
    - Un diccionario con la similitud media del coseno para cada cluster.
    """
    cluster_labels = modelo_kmeans.labels_
    unique_labels = np.unique(cluster_labels)
    cos_sim_cluster = {}
    
    for label in unique_labels:
        # Seleccionar características del cluster actual
        cluster_features = features_reducidas[cluster_labels == label]
        # Calcular similitud del coseno dentro del cluster
        cos_sim = cosine_similarity(cluster_features)
        # Opcional: promedio de similitudes excluyendo la diagonal principal
        np.fill_diagonal(cos_sim, 0)
        cos_sim_mean = np.mean(cos_sim[cos_sim != 0])
        cos_sim_cluster[label] = cos_sim_mean
    
    return cos_sim_cluster

