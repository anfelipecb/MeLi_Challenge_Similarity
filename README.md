<img src = "./Assets/imgs/ML-Codo-g.jpg" alt = "Encabezado MLDS" width = "80%" style="float: left; margin-right:20px" >  </img>

## **MeLi Challenge - Similitud de items**
### **archivo README.md**
**Por: Andrés Felipe Camacho Baquero - [Personal website](https://anfelipecb.github.io/)**

--- 
En este challenge nos piden lo siguiente: 
    Dentro del marketplace existen productos similares o idénticos entre sí (son productos vendidos por distintos sellers, en la api puedes obtener y descargar los títulos e incluso las imágenes!). ¿Cómo buscar dichos ítems para agruparlos y volverlos comparables entre sí? Esto permitiría mejorar la experiencia ante muchas opciones similares.

---

Los notebooks aquí se manejan de manera anidada. Aquí un resumen 
- **Notebook00** es el analisis exploratorio inicial, es desordenado pero corre continuamente. El proceso logid¿co para llegar a mi solución final 
- **Notebook01** Se extraen los datos de la API adecuadamente 
- **Notebook02** Se extraen los embeddings de texto y de imagen 
- **Notebook03** se agrupan los superembeddings obtenidos mediante clusters y se descargan las imagenes


### Antes de empezar 

Debemos clonar el respositorio adecuadamente. Pasos para clonarlo: 

1. Desde la terminal, escoger el directorio local donde se quiere clonar el repositorio (ej: 'cd Users/anfelipecb/MeLI/Proyectos')
2. Clonar el repositorio usando git clone: 
```
git clone https://github.com/anfelipecb/MeLi_Challenge_Similarity.git
```
3. Navegar y asignar al repositorio creado:
```
cd tu-repo-configurado
```
4. Desde la terminal del proyecto, creamos y activamos el entorno virtual (acá lo llamamos melienv):
```
python3 -m venv melienv
source myenv/bin/activate
```
5. Para agregar el entorno virtual a jupyter primero nos aseguramos de instalar el paquete ipykernel que nos permitirá ejecutar python desde jupyter dentro del entorno virtual: 
```
    `pip install ipykernel`
```
6. Registramos el el entorno virtual como kernel a Jupyter
```
    `python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"`
```

## Configuración del proyecto: Para hacerlo automáticamente: 
(Se ha creado un script con la instalación y configuración, llamado  'setup.sh')
    Antes de ejecutarlo, debe hacerlo ejecutable. Abrimos la terminal donde está setup.sh
        `chmod +x setup.sh`

Para configurar el proyecto automáticamente, siga estos pasos:

1. Abra la Terminal.
2. Navegue hasta el directorio donde desea clonar el repositorio.
3. Ejecute el siguiente comando:

```bash
./path/to/setup.sh
```

Todas las dependencias usadas en los notebooks están resumidas en el archivo "requirements.txt" 
    Para asegurarnos que tenemos todo lo que se necesita listado en este archivo, desde el entorno activado corremos: 
        `pip install -r requirements.txt`


#### Este es el arbol del proyecto, que obtuve con el comando tree (sobre la terminal). Excluto las imagenes finales:

```
tree  -I  'melienv'

```

```
├── Assets
├── Notebooks
│   ├── 00_Exploracion API y Datos.ipynb
│   ├── 01_Analisis de datos.ipynb
│   ├── 02_Extraccion_Embeddings.ipynb
│   ├── 03_AgrupacionyAnalisis.ipynb
│   └── figures
│       └── treemap_categorias2.png
├── README.md
├── data
│   ├── final
│   │   ├── combined_f_Res50.npy
│   │   ├── combined_f_efficientb0.npy
│   │   ├── combined_f_efficientb7.npy
│   │   ├── combined_f_xception.npy
│   │   ├── data_items_all.csv
│   │   └── ids_efficientnetb7.npy
│   ├── processed
│   │   ├── PCA_ResNet50.png
│   │   ├── PCA_ResNet50_2D.png
│   │   ├── PCA_ResNet50_3D.png
│   │   ├── tSNE_ResNet50_2D.png
│   │   ├── tSNE_ResNet50_3D.png
│   │   ├── tSNE_efficientb0_2D.png
│   │   ├── tSNE_efficientb7_2D.png
│   │   ├── tSNE_xception_2D.png
│   │   └── text_tSNE_BERT_2D.png
│   └── raw
│       ├── df_items_especificos.csv
│       ├── df_solo_audifonos.csv
│       └── images
├── environment.yml
├── scripts
├── setup.sh
└── similarity
    ├── setup.py
    └── similarity
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-311.pyc
        │   ├── config.cpython-311.pyc
        │   └── custom_funcs.cpython-311.pyc
        ├── config.py
        └── custom_funcs.py
```