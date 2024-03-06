import requests
from .config import API_URL

def obtener_items_por_categoria(cat_id, offset=0):
    """
    Función para obtener items de una categoría específica desde la API de MercadoLibre.
    
    Parámetros:
    - cat_id: ID de la categoría a consultar.
    - offset: Desplazamiento en los resultados de la búsqueda (para paginación).
    
    Retorna:
    - Un objeto JSON con los items encontrados.
    """
    # Construir la URL para la búsqueda de categorías
    url_categorias = f'{API_URL}/sites/MLA/categories'
    
    # Opcionalmente, puedes verificar las categorías primero
    # respuesta_cats = requests.get(url_categorias)
    # categorias = respuesta_cats.json()
    
    # Construir la URL para la búsqueda de items en una categoría con desplazamiento específico
    url_items = f'{API_URL}/sites/MLA/search?category={cat_id}&offset={offset}'
    
    # Hacer la solicitud para obtener los items
    respuesta_items = requests.get(url_items)
    if respuesta_items.status_code == 200:
        items = respuesta_items.json()  # Convertir la respuesta en JSON
        return items
    else:
        # Manejo básico de errores: retorna None si algo va mal
        return None