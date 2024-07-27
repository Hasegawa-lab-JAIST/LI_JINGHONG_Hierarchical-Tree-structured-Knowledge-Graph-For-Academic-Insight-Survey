# Hierarchical Tree-structured Knowledge Graph:Gain an Insight View for Specific research Topic (On building, plan to finish uploading the notebook in July,  pyfile and app in Sep)


# 0. About this 
## Target
- 1. Create a knowledge interface for insights survey assistants from multiple academic papers on a specific research topic. 
- 2. From the origin (root) of the research task, expand the citation inheritance and relevance associations. 
- 3. Explore the relevance chain within similar research tasks to highlight key research points.

## Resources
- **Link of our preprint :** 
	- [Arxiv](https://arxiv.org/abs/2402.04854)
	- [ResearchGate](https://www.researchgate.net/publication/378339313_Hierarchical_Tree-structured_Knowledge_Graph_For_Academic_Insight_Survey)

- **Link of our dataset :**
	- [https://doi.org/10.34740/KAGGLE/DS/4330260](https://doi.org/10.34740/KAGGLE/DS/4330260)

## Citation (Current)
```
Li, J., Phan, H., Gu, W., Ota, K., & Hasegawa, S. (2024). 
Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey. 
arXiv preprint arXiv:2402.04854.
```
	
## Declaration 
**The article of this project has been accepted by "THE 18TH INTERNATIONAL CONFERENCE ON INNOVATIONS IN INTELLIGENT SYSTEMS AND APPLICATIONS (INISTA 2024)", and future citations should be based on the official version.**

# 1. Content 
## Overview
<img width="1868" alt="Figure1-2" src="https://github.com/user-attachments/assets/8cb8914f-925b-4edc-a7ca-22c8669c7315">

## Phase1 - Text-Processing.ipynb 
- Perform section & sentence segmentation on the total dataset of ‘HotpotQA’ topic ([Hotpotqa_paper_content.csv](https://doi.org/10.34740/KAGGLE/DS/4330260)) to create the corresponding insight sentence dataset. Experts then assign insight labels('Issue Resolved', 'Neutral', and 'Issue Finding' ) to this dataset.

## Phase1* - Cite-Net-Construction.ipynb
- The cite-net, established based on the citation relationships among papers in the Hotpotqa topic, is presented in Cite_net_total.html as the overall network. The sub-nets within it will be extracted to form the basic prototype of the tree-structured knowledge graph.


## Phase2 - Insight-Sentence-Classification.ipynb
- This document implements a classifier for Insight sentences. All insight sentences will be classified into 'Issue Resolved', 'Neutral', and 'Issue Finding' categories using machine learning methods. The classification results will be used to establish the relevance chain in the graph.

## Phase3&4 - Hierarchical-tree-building-vis.ipynb
- Code for building & visualizing the graph

### Phase3 - Similarity-Calculation.ipynb
- Calculate the similarity of the elements of the insight section & relevance chain (‘Issue finding’ → ‘Issue Resolved’) in the paper-network.


### Phase3,4-I - For Inheritance tree: Hierarchical-inheritance-tree-building&vis.ipynb

### Phase3,4-R - For Relevance tree: Hierarchical-Relevance-tree-building.ipynb

## Appendix - Freq-dis.ipynb
- This document displays the frequent words in the insight section (conclusion, discussion, limitation).

## Project
## App


# 2. Environment
```python
```

# 3. Preparation
- Extract content of papers containing the keyword 'HotpotQA' from the S2ORC dataset.
- The data of content of papers is in the file `Hotpotqa_paper_content.csv` in [https://doi.org/10.34740/KAGGLE/DS/4330260](https://doi.org/10.34740/KAGGLE/DS/4330260)


# 4. Example
- [Inheritance tree](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/Output/Inheritance_tree.html)

- [Relevance tree](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/Output/Inheritance_tree.html)

# To Do
- Prompt enginnering for abstractive summarization.
- User Interface development.
- Adjust the appearance of the graph.

# Acknowledgements
This work was supported by JSPS KAKENHI Grant Number JP20H04295.

# Notes 

If you utilize any resources (Concepts, code, or methods) from this repository, please cite our paper and acknowledge the source code & dataset.

# License
This repository is released under the Apache 2.0 license as found in the [License](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/LICENSE) file.


