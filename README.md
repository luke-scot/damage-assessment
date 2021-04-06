# MRes damage assessment project outline

## Objective
* Infer damage rapidly from sparse and various data sources available immediately following disaster
    * Train model on Beirut to predict damage from August 2020 explosion
    * 

## Data Sources
* Building footprint maps - [OSM](https://www.openstreetmap.org/export#map=15/33.8994/35.5006)
* Beirut building data - https://beirutrecovery.org/ , [armyGIS](https://gis.army.gov.lb/lm/index.php/view/map/?repository=15&project=open)
    * Damage Assessments
    * Building details (sparse)
* Before/After high res imagery - [Beirut example](https://beirutrecovery.org/) - do we have access?
* Before/After Sentinel Imagery - Descartes Labs?
* Ground imagery - [mapillary](https://www.mapillary.com/app/user/lshc3?pKey=3_4f3JT6cNUvSEhkWyc8wg&lat=33.901549420449854&lng=35.490283832833825&z=17.637259530296614)
* Social Media Data

## Methods
* Build-graph representation of buildings in city
* Propagate belief of damage extent within graph - [NetConf](https://github.com/dhivyaeswaran/dhivyaeswaran.github.io/tree/master/code)
    * Uncertainty decomposition within the graph allowing us to combine benefits of multiple data sources

## Week 1 tasks
1. Get datasets into useable formats
2. Visualise datasets in jupyter notebook
3. Convert NetConf to Python