[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/luke-scot/damage-assessment.git/HEAD?filepath=colab_demo.ipynb)

# MRes damage assessment project outline

## Objective
* Infer damage rapidly from sparse and various data sources available immediately following disaster - update beliefs as more data becomes available
    * Train model on Beirut to predict damage from August 2020 explosion
    * If successful on training location - apply model to new locations
    * If transferable - pre-emptively obtain building data for exposed cities to enable rapid assessment following disaster

## Data Sources
* Building footprint maps - [OSM](https://www.openstreetmap.org/export#map=15/33.8994/35.5006)
* Beirut building data - [beirutrecovery.org](https://beirutrecovery.org/), [armyGIS](https://gis.army.gov.lb/lm/index.php/view/map/?repository=15&project=open), GeoPal app
    * Damage Assessments
    * Building details
* Ground imagery - [mapillary](https://www.mapillary.com/app/user/lshc3?pKey=3_4f3JT6cNUvSEhkWyc8wg&lat=33.901549420449854&lng=35.490283832833825&z=17.637259530296614)
* Before/After high res imagery - [Beirut example](https://beirutrecovery.org/) - ask for access from WorldView & Copenicus EMS?
* Before/After Sentinel Imagery - Descartes Labs or pre-processed from SNAP?
* Social Media Data - geolocated posts from aftermath?

## Methods
* Build graph representation of buildings in city
    * Nodes - Pixels to begin with (~ Sentinel resolution)
        *  Future - maybe individual buildings? city blocks?
    * Edges - Based on geographical locations 
        * Future - Based on building types? hazard exposure (e.g. along fault line for earthquakes)?
    * Weights - Based on similarity of pixels
* Propagate belief of damage extent within graph - [NetConf](https://github.com/dhivyaeswaran/dhivyaeswaran.github.io/tree/master/code) by [Eswaran et al., 2017](https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.17)
    * Beliefs of damage decision from GeoPal (Green, Amber, Red)
        * Alternative - damage rating (e.g. no dmg, minor dmg, moderate dmg, major dmg, destroyed)?
    * Belief values based on source of info (e.g. high to low: ground assessment, ground image, satellite high-res image, satellite low-res, ...)
        * Ground Assessment - Very high belief for labels of nodes within assessed building
        * Satellite SAR/Imagery - Pixel values from single measurement or amplitude change between pre/post disaster measurements? Lower beliefs than ground truth. Beliefs dependent on data quality and label confidence.
        * Ground imagery - Label prediction from output of CNN, beliefs for each label for node from output probabilities of CNN recognising damage?
        * Possible -> Social Media Data - label prediction from output of NLP on geolocated posts?
    * Uncertainty decomposition (~multi-source uncertainty framework, [Zhao et al., 2020](https://arxiv.org/pdf/2010.12783.pdf)) within the graph will allow us to combine benefits of multiple data sources
* Any consideration of likely number of people affected to focus S&R? Or leave that judgement to S&R crews? - Maybe further work
## Week 1 tasks
1. Get datasets into useable formats
2. Visualise datasets in jupyter notebook
3. Convert NetConf to Python and to accept beliefs more than 2 classes?
