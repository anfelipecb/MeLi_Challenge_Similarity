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

# Registrar el entorno virtual como kernel en Jupyter
python -m ipykernel install --user --name=melienv --display-name="Python (melienv)"

# Instalar otras dependencias desde requirements.txt si existe
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "Entorno configurado y activado. Kernel de Jupyter listo para usar."
