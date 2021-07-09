# Graph-Based Belief Propagation for Post-Disaster Damage Assessment

[![hackmd-github-sync-badge](https://hackmd.io/OrUFc3xLT1yftbxL0k5QZg/badge)](https://hackmd.io/OrUFc3xLT1yftbxL0k5QZg)


## Introduction

### Application Background

In the aftermath of a major disaster, such as the 2015 Nepal earthquake or the 2020 Beirut explosion, the Local Emergency Management Agency (LEMA), generally a designated branch of the local government, is responsible for coordinating the response. Oftentimes, a request for international aid is directed to the United Nations' (UN) [Inter-Agency Standing Committee (IASC)](https://interagencystandingcommittee.org/) who co-ordinate the deployment of international teams from UN agencies (e.g. [UN Disaster Assessment and Coordination (UNDAC)](https://www.unocha.org/our-work/coordination/un-disaster-assessment-and-coordination-undac)), NGOs (e.g. [SARAID](https://www.saraid.org/)) and international governments to the affected area. The scope of work of international teams with expertise in rescue operations and engineering, is dictated by the LEMA according to their needs and falls under two priority activities which we will address in turn: Search and Rescue (S&R) and Damage Assessment (DA).

#### Search and Rescue

Post-disaster a rapid S&R response can significantly reduce casualties due to infrastructure damage, therefore co-ordinated and effective organisation is crucial to avoid missed areas or duplication. The first task of S&R is sectorisation, see figure 1, with number and types of reconaissance teams deployed according to a sector's likely needs. Sector prioritisation is a crucial response step to reach the most affected areas quickly and save lives.

Figure 1: Sectorisation - to divide the affected area into sectors according to geography, population density, building vulnerability with consideration given to likely gatherings or damages to key infrastructure (e.g. hospitals), Beirut 2020.

Decision making during prioritisation must be justifiable, should there be a later inquiry, based upon prior knowledge of an area's vulnerability and initial reports received through remaining communication channels or aerial photography. However a systematic approach to collating the sparse and inconsistent data available does not exist. Herein lies an opportunity to create a novel framework for combining data from multiple sources, including structural assessments and satellite imagery, as it become available to infer a rapid initial damage assessment. Associated with a quantification of uncertainty in damage estimates, such a tool would be a very useful for prioritising resource allocation.

#### Damage Assessment

A more overlooked part of disaster response than S&R is damage assessment. DA is key for preventing avoidable casualties caused by structural failures occurring in the hours or days post-disaster. Furthermore a disaster does not end when casualties are accounted for, livelihoods must be restored without delay to avoid a disaster triggering a humanitarian crisis which could last years or even generations as individuals and families are displaced or left in ruin. DA is the first step to recovery by determining priorities for structural intervention and for direct humanitarian aid to those most in need. 

Under the leadership of UNDAC a consistent approach to building damage classification is being developed based on the [Applied Technology Council](https://www.atcouncil.org/)'s [methodology](http://www.atcouncil.org/pdfs/ATC45Rapid.pdf) using a traffic light system to label buildings in a range from safe to unsafe. Firstly, as with S&R, a systematic approach to rapidly inferring a building's classification would be of great use in directing engineering and humanitarian response. Secondly, a consistent mapping of DA with increasingly available ground information will aid in communicating the extent of damages and actions required (e.g. evacuate building) to residents and responders alike.

The increasing digitilisation of response practices, such as structural assessments, photography and geotagged social media posts further increases the need for a flexible framework to combine and make use of all the data available in real-time.

<!-- % Think about techonology progression (more and more interactive)
% Figure for UNDAC classification?
-->


### Modelling Background

#### Model Requirements

Table 1 summarises the requirements of a model able to synthesise the available data into an actionable output for DA. 

| Properties    | Requirements | 
| ------------- | ------------ |
| Input    | Multi-Modal Data - Many different types <br> Sparse Datasets - Incomplete with many gaps <br> Frequently Updated |
| Output        | Uncertainty Quantification - Scale of confidence in prediction<br> Maximise Data Exploitation - Use as much data as available <br> Easily Interpretable - Understandable to anyone |
| Computation   | Flexible - Various input types <br> Rapid - Runs in seconds to minutes <br> Scalable - Usable in areas of different sizes <br> Inexpensive - Runs anywhere in real time |

 

#### Graph-Based Belief Propagation 

A graph-based approach satisfies the input requirements stated in table 1 by allowing multi-modal data to be assigned to individual nodes each representing a small patch of the damaged area such as a 10x10 metre square. The damage probability for each node can then be updated as additional input data becomes available.

Belief Propagation (BP) within the graph structure allows for inference of DA based on the similarities between nodes. Figure 2 shows how edges are constructed between similar nodes according to their properties from input datasets or simply geographical proximity. Beyond simple label-propagation our requirement for uncertainty quantification lead us to a model incorporating belief propagation (BP) to account for uncertainty in the labelling of each node.

Figure 2 - Graph representation basic graph

Together with the computational considerations set out in table 1, our requirements lead us to constructing our model based on the Dirichlet-Multinomial BP algorithm called NetConf created by Eswaran et al., 2017, with a run-time scaling linearly with an increase in nodes.


### Previous Work

Applying NetConf as a multi-modal BP approach to damage assessment builds upon work in 3 related fields. 

#### Remote Sensing for Damage Assessments
Remote sensing in the broadest sense includes any measurement taken at a distance from the measured object. In the context of this study we will narrow the focus of remote sensing to satellite measurements of the Earth's surface within the area of interest. Satellite observations are the only way to gain a complete image of the area soon after a disaster, and have the benefit of being generally independent of a countries resources with global coverage. 

The first type of satellite measurements is passive observation which senses radiation in the visible and near infrared regions of the electromagnetic spectrum \cite{IntrotoRS}. Visible imagery is the most easily interpretable due to the familiarity of this part of the spectrum to a human interpreter. Studies including ... and ... have used imagery to identify damages through segmentation and ... respectively. The limitations of using visible imagery are threefold. Firstly High-Resolution (HR) imagery (<5 metres per pixel) is not readily available and resolutions can still be too low to identify any detailed damages. Secondly, imagery is taken from directly above the area and damage will only be identified if the rooftops or parts of the infrastructure nearest the satellite have changed their radiative properties. Finally, as passive measurements rely on solar illumination of the Earth's surface, they are sensitive to time-of-day and inclement weather, limiting the availability of high quality data.

Another type of 

* 

#### Belief Propagation for Remote Sensing

#### NetConf for Belief Propagation

### Study Outline


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
