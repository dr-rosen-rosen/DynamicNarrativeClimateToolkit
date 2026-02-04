## dynClimToolkit
Dynamic Climate Toolkit

# Set up
Create a venv with the requirements.txt file, using python 3.9
    python3.9 -m venv .dynClimVenv
    
    source .dynClimVenv/bin/activate
    
    pip install -r requirements.txt

    pip install -e . # This links local dev packages to env

# set up ipykernel
    python -m ipykernel install --user --name=dynClimKern --display-name="Dynamic Climate Kernel"

# install spacy models
    python -m spacy download en_core_web_sm

# IF using DB options
# Create a .env file with this info:
# options for db engine
DB_TYPE=
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=