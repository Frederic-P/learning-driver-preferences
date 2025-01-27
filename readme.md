# Learning driver preferences 

## Virtual environment management: 

### Installation
VENV-management is handled by Anaconda. Import the file into anaconda with the following steps: 
- open anaconda
- click the "Environments" tab
- Click import > choose the local drive option in the popup window
- Navigate to venv.yaml in this repo and load it
- (optional): Give your Venv a descriptive name
- Press import (wait)


## choochoo: The train of thought. 

VRP probleem is op te lossen met een python package als: 
- https://vrpy.readthedocs.io/en/latest/
- https://github.com/yorak/VeRyPy
- https://developers.google.com/optimization/routing/original_cp_solver 
(redelijk wat opties als je zoekt, weet niet als alles in Conda zit, maar kun je omheen.)

Mochten we de echte coordinaten gebruiken, kunnen we OSM combineren met NEO4J - kunnen ons baseren op:
- https://neo4j.com/developer-blog/routing-web-app-neo4j-openstreetmap-leafletjs/ 
    - NEO4J is installeerbaar via DOCKER - niet zo gek moeilijk. 
    - https://www.youtube.com/live/Z4XZgsbaD9c 
- 