# MeLi_Challenge_Similarity

Por: Andrés Felipe Camacho
Personal-site: [anfelipecb.github.io](https://anfelipecb.github.io/)


Este código fue creado en 

Debemos clonar el respositorio adecuadamente. Pasos para clonarlo: 

1. Desde la terminal, escoger el directorio local donde se quiere clonar el repositorio (ej: 'cd Users/anfelipecb/MeLI/Proyectos')
2. Clonar el repositorio usando git clone: 
```
git clone https://github.com/anfelipecb/MeLi_Challenge_Similarity.git
```
3. Navegar y asignar al repositorio creado:
    `cd tu-repo-configurado`

4. Desde la terminal del proyecto, creamos y activamos el entorno virtual (acá lo llamamos melienv):
    `python3 -m venv melienv`
    `source myenv/bin/activate`
5. Para agregar el entorno virtual a jupyter primero nos aseguramos de instalar el paquete ipykernel que nos permitirá ejecutar python desde jupyter dentro del entorno virtual: 
    `pip install ipykernel`
6. Registramos el el entorno virtual como kernel a Jupyter
    `python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"`
(Se ha creado un script con la instalación y configuración, llamado  'setup.sh')
    Antes de ejecutarlo, debe hacerlo ejecutable. Abrimos la terminal donde está setup.sh
        `chmod +x setup.sh`
    Ahora sí se puede ejecutar: 
        `./setup.sh`

Todas las dependencias usadas en los notebooks están resumidas en el archivo "requirements.txt" 
    Para asegurarnos que tenemos todo lo que se necesita listado en este archivo, desde el entorno activado corremos: 
        `pip install -r requirements.txt`


Corremos el código en 'Connection_API.ipynb'
