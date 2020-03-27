conda create --name neural_modelling python=3.6 --file conda-spec-file.txt
conda activate neural_modelling
conda install -n neural_modelling pip
conda install torchvision
pip install -r requirements.txt
