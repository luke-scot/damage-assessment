# Graph-Based Belief Propagation for Post-Disaster Damage Assessment

## Introduction

### Background

#### Event Aftermath & mobilisation – immediate aftermath of a disaster (natural such as Haiti earthquake or unnatural such as Beirut explosion)
* Local governments (ministries vary between countries) coordinate disaster response often aided by UN agencies under the umbrella of https://interagencystandingcommittee.org/ which coordinate cooperation between different UN agencies. Known as LEMA (Local Emergency Management Agency).
* Depending on local resources and needs due to disaster this organisation often request aid from international teams which come to assist and take pressure off. Can be in form of NGOs such as SARAID, or foreign governments who may take longer to obtain permissions due to political relations
* Whether the LEMA take charge of response management or it is requested of a response team the first step is a prioritisation of S&R targets.
* Organisation is crucial to avoid duplication and missed areas
* https://www.insarag.org/methodology/insarag-guidelines/


#### S&R Prioritisation
  * Sectorisation – The first element to an S&R operation is sectorisation of a city or area dependent on its geography. Sectors are established according to area type and vulnerability with consideration given to likely gatherings and dense population areas or facilities. These should initially be of a reasonable size to be allocated to a reconnaissance team.
  * Prioritisation of sectors – One of the most crucial elements to an effective rescue operation is prioritisation of sectors. Decision making (which must be justifiable should there be a later inquiry) is made on the basis of both prior knowledge of the area (vulnerable buildings, population centres) and initial reports (comms or aerial photography – MapAction). There is no systematic approach to assessing vulnerability and knowledge of damages and risks. A rapid indication of most damaged areas would be a very useful additional tool in prioritising reconnaissance. Furthermore ground truthing from this reconnaissance which is becoming more digital as years progress could feed back into model to improve predictions.
  * There are further considerations at this point, such as survivability, for example the sometimes difficult decision to prioritise areas where there is higher chance of survival than for example in the immediate vicinity of an explosion.
  * Reconaissance
  
#### Damage Assessment (DA)
  * A more overlooked part of disaster response than search and rescue is damage assessment. 
  * DA is key for to preventing avoidable casualties from building damages or collapses that occur in the following hours/days. And also for beginning the route to recovery and humanitarian needs. A disaster does not end when the casualties are all accounted for, livelihoods and ways of life must be restored without delay to avoid after effects of disaster s which can last years, even generations as individuals and entire families can be dislodged or lose livelihoods.
  * Structural assessments determine prioritisation of demolition/reconstruction/reinforcements – DAC procedure of Green/Amber/Red. If this process could be sped up and inferred that would be very useful.
 

#### Graph-Based Belief Propagation

* Why graph-based?
    * In an emergency response, available data varies in type and completeness. Therefore we must build a framework capable of incorporating whatever varied and sparse data is immediately available with the potential to build upon the model outcome as more data becomes available.
    * The nodes in a graph allow for multi-modal data to be incorporated into a single framework for estimating damage. These nodes are associated with each other along edges which can be attributed according to any number of different data sources.

### Previous Work
* BP for remote sensing
* DLR's published damage assessment
* DA from UN-Humanitarian but that involves going around every house

### Project outline



## Methods
### Graph Representation

### NetConf

### Data

## Results
### Curated Houston Dataset

### Beirut 


## Discussion
### Effectiveness of the method
* Houston data shows that the method can work but only for sufficiently different classes
* Comment on effectiveness of each type of data
    * Some methods are much more effective than others. Additional data sets never hurt but can provide large quantity of help. 
    * Geographical edges are not always most helpful

### Application to Beirut
* Problems with heavy bias towards green listed buildings over large area
    * When focussed in smaller area with sampling of a better distribution. Classification is shown to be effective.
    * Belief propagation works with a certain effectiveness but there are problems in being particularly certain. However as seen with Houston this may be improved with more data.

### Applicability of technique
* The framework built up is very interoperable and can be adjusted to incorporate any amount of new data types for new locations.

## Conclusion
* Despite limited effectiveness in Beirut thus far, the platform built is an excellent starting point upon which to build further research. The functioning on Houston along with previous studies show that this technique can be effective in this domain with additional work.

### Future work
* Try different BP algorithms
* Incorporate more data sources
* More effective or intelligent sampling of high-resolution data
