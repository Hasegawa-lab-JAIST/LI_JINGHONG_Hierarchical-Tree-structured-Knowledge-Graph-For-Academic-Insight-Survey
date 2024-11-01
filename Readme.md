# Hierarchical Tree-structured Knowledge Graph:Gain an Insight View for Specific research Topic (On building, plan to finish uploading the notebook in July,  pyfile and app in Oct)


# 0. About this 
## Target
- 1. Create a knowledge interface for insights survey assistants from multiple academic papers on a specific research topic. 
- 2. From the origin (root) of the research task, expand the citation inheritance and relevance associations. 
- 3. Explore the relevance chain within similar research tasks to highlight key research points.

## Resources
- **Link of our paper :**
  	- [https://ieeexplore.ieee.org/document/10683856](https://ieeexplore.ieee.org/document/10683856)

- **Link of our dataset :**
	- [https://doi.org/10.34740/KAGGLE/DS/4330260](https://doi.org/10.34740/KAGGLE/DS/4330260)

## Declaration 
**The article of this project has been published by "THE 18TH INTERNATIONAL CONFERENCE ON INNOVATIONS IN INTELLIGENT SYSTEMS AND APPLICATIONS (INISTA 2024)."**

## Citation 
```
J. Li, P. Huy, W. Gu, K. Ota and S. Hasegawa, "Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey," 2024 International Conference on INnovations in Intelligent SysTems and Applications (INISTA), Craiova, Romania, 2024, pp. 1-7, doi: 10.1109/INISTA62901.2024.10683856.
```

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
- Code for tree-building algorithm & Graph
Visualization 
- To maintain consistency of input and output, the code for cite-net construction and similarity calculation has also been added.

## Appendix.1 - Freq-dis.ipynb
- This document displays the frequent words in the insight section (conclusion, discussion, limitation).

## Appendix.2 - Similarity-Calculation.ipynb
- Calculate the similarity of the elements of the insight section & relevance chain (‘Issue finding’ → ‘Issue Resolved’) in the paper-network.

## Project
+ [ ] To do



# 2. Preparation
- Extract content of papers containing the keyword 'HotpotQA' from the S2ORC dataset.
- The data of content of papers is in the file `Hotpotqa_paper_content.csv` in [https://doi.org/10.34740/KAGGLE/DS/4330260](https://doi.org/10.34740/KAGGLE/DS/4330260)


# 3. Example(HTML) 
- [Inheritance tree](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/Output/Inheritance_tree.html)

- [Relevance tree](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/Output/Inheritance_tree.html)

# 4. User Interface
- Current
<img width="946" alt="スクリーンショット 2024-09-26 20 22 27" src="https://github.com/user-attachments/assets/4fdb4647-6166-46da-a9af-e3c3d9012480">


+ [ ] To do 1: Improve layout of components

# Acknowledgements
This work was supported by JSPS KAKENHI Grant Number JP20H04295.

# Notes 

If you utilize any resources (Concepts, code, or methods) from this repository, please cite our paper and acknowledge the source code & dataset.

# License
This repository is released under the Apache 2.0 license as found in the [License](https://github.com/Hasegawa-lab-JAIST/LI_JINGHONG_Hierarchical-Tree-structured-Knowledge-Graph-For-Academic-Insight-Survey/blob/main/LICENSE) file.


