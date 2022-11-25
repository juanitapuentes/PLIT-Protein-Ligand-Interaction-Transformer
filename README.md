# PLIT: Protein-Ligand Interaction Transformer


With the aim of identifying new ligands that can be used for the Drug Discovery process we propose PLIT (Protein-Ligand Interaction Transformer). PLIT proposes the use of graph representations based on SMILES to exploit the information of ligands to predict the interaction with proteins. PLIT consists of a Deep Graph Convolutional Network that considers the ligands atoms’ and bonds’ most important features. Moreover, it includes a Transformer Encoder that uses its self-attention mechanism to encode the information of the graphs feature representation of each ligand.




# PLIT Architecture
The GCN module is used to extract structural and spatial information from the graph of the ligands. The representation obtained from the GCN is optimal for PLIs analysis. The GCN module performs first the normalization and ReLU activation, followed by the graph convolution and the addition of the residual connection. Our GCN module consists of 20 message-passing layers. The final embedding size for nodes’ (atoms) and edges’ (links) features of 128. The updated graph obtained after the GCN module contains the 128-D or 256-D features. This structure is fed into a Transformer encoder, which consists of two main components: a self-attention mechanism and a feed-forward neural network. The output of the Transformer encoder is received as input of a Convolutional Neural Network (CNN) with different number of layers. Finally, the output vector of the CNN is used as input to a linear layer that calculates the associated probability of the ligand to interact with each of the 102 proteins and classifies the ligand to the class (protein) with the highest probability.

**PLIT  outperforms State-of-the-art in the proposed benchmark**

<img width="905" alt="arquitectura" src="https://user-images.githubusercontent.com/98660892/204027480-dbcb5662-8de4-48e6-929c-3f12ee08f927.png">

# Set Up Environment

The following steps are required in order to run PLIT:
```
conda create --name PLIT
conda activate PLIT

bash env.sh
```
# Repository Structure

Below are the main directories in the repository: 

- `data/`: the data loader, feature extraction scripts and datasets
- `model/`: models definition, along with the hyperparameter settings used for experimentation
- `utils/`: args for running train and inference. Checkpoints saving files and scripts for metrics calculation

Below are the main executable scripts in the repository:

- `main.py`: main training and validation script
- `ensamble.py`: calculates metrics on the test set

