#!/bin/zsh

# Clonar el repositorio
git clone https://github.com/anfelipecb/MeLi_Challenge_Similarity.git
cd MeLi_Challenge_Similarity

# Crear y activar el entorno virtual
python3 -m venv melienv
source melienv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install ipykernel
pip install "nbformat>=4.2.0"
pip install ipywidgets
pip install -U kaleido
pip install -U efficientnet
pip install -U "scikit-learn"
pip install transformers
pip3 install torch torchvision
# Registrar el entorno virtual como kernel en Jupyter
python -m ipykernel install --user --name=melienv --display-name="Python (melienv)"

# Instalar otras dependencias desde requirements.txt si existe
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Instalar dependencias desde setup.py
# Asegúrate de que setup.py esté configurado para instalar todas las dependencias necesarias
pip install -e .

echo "Entorno configurado y activado. Kernel de Jupyter listo para usar."

jupyter nbextension enable --py widgetsnbextension
