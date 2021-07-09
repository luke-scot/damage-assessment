# Graph-based belief propagation for post-disaster damage assessment

Check out the [demo on Google Colab](https://colab.research.google.com/github/luke-scot/damage-assessment/blob/main/colab_demo.ipynb)! 

## Abstract

Disaster response teams allocate resources for search-and-rescue and humanitarian aid
based on infrastructure damage assessments. Complete ground-based assessment is too
slow for an eﬀective response, therefore, decisions are based on inconsistent initial re-
ports and, if available, qualitative observation of satellite imagery. To systematise disaster
response prioritisation, I have created a framework that learns from the disparate data
available to rapidly infer a damage map. By extending Eswaran et al.’s (2017a) NetConf
algorithm, my model combines remote sensing and ground data sources into a multi-
modal graph representation of the disaster location, through which damage beliefs are
propagated. The resulting conﬁdence-aware damage map quantiﬁes uncertainty in the
predictions which is vital for decision-making. Applying NetConf to spatial data exposes
limitations in its scalability, however, when tested in an idealised land classiﬁcation sce-
nario, performance reaches a very high F1 score of 0.9. Applied to the real-world damage
assessment following the 2020 Beirut port explosion, the model requires balanced class rep-
resentationsand35%groundtruthdatabeforesurpassinganF1scoreof0.6. Thereforethe
framework is currently unsuitable for immediate aftermath search-and-rescue, but rising
performance as data availability increases renders it a useful tool for accelerating damage
assessment, and hence the recovery phase of disaster response. The framework is pur-
posely ﬂexible, built to propagate beliefs while quantifying uncertainty for any large scale
multimodalapplication, notlimitedtodamageassessmentorremotesensing. Forademon-
stration please visit https://colab.research.google.com/github/luke-scot/damage-
assessment/blob/main/colab_demo.ipynb. Alternatively, the complete model is avail-
able at https://github.com/luke-scot/damage-assessment.
