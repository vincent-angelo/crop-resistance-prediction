# Crop Resistance Predictor
Vincent Angelo, Luis Chan, Justin Lam

## Introduction
### Overview
In agriculture, the early and accurate prediction of disease resistance in crops is necessary for stable food
production. Traditional methods of breeding and disease testing are time-consuming and labor-intensive.
Machine learning has the potential to improve both the efficiency and efficacy of disease resistance prediction
by taking advantage of the increasing availability of genomic data.

Previous studies have applied models such as support vector machines, decision trees, and multilayer percep-
trons to this problem [6]. However, these models are relatively simple and can fail to capture the interactions
and dependencies between genetic markers, which can play a critical role in disease resistance.
We addressed this by applying a graph neural network (GNN) to the problem. Specifically, we
used a Graph Attention Network (GAT) to predict disease resistance from single-nucleotide polymorphisms
(SNPs). SNPs are changes to a single genetic base in DNA, which can impact the resulting biology of the
plant. This is a regression problem, where we attempt to predict the infection spread and severity in plants
from SNP data.


## Data
### Winter Wheat Stripe Rust Resistance

This dataset comprises the SNPs responsible for different severities of Stripe Rust infection in winter
wheat. The goal of this project was to use the data to predict the infection type and area through regression.
The data is publicly available, and has been curated from the National Small Grains Collection (NSGC)
winter wheat germplasm collection, totalling 1175 accessions, with 127 Quantitative Trait Loci (QTL)-
SNPs. The different accessions’ origin country is available, along with the genetic structure subgroup [1].
The dataset is complete with the chromosome number and location of the SNPs of interest and the frequency
of favorable alleles.
Of particular interest are Infection Type (BLUE-IT) and Severity (BLUE-SEV). BLUE-IT represents the
intensity of infection and ranges from 0 to 9, where 0 indicates no sporulation (i.e., the winter wheat stripe
rust infection has not produced spores) and 9 indicates sporulation and necrosis (i.e,. tissue death). On the
other hand, BLUE-SEV is the percentage of the leaf that has been infected, ranging from 0 to 100. These
are the ground truth labels for our regression task.
However, preprocessing is necessary to get the data into the required format and structure for our model
architecture, which we will expand on in the next section.


## Methodology
### Feature Engineering

We chose the graphical lasso method to build the graph for our model to learn on, while simultaneously
selecting for only a subset of SNPs that have the most influence on each other.
The graphical lasso is a sparse estimator for the precision matrix Θ (which is the inverse of the covariance matrix) of the data.

$$
\hat{\Theta} = \mathop{\mathrm{argmin}}\limits_{\Theta \succeq 0} \left( \mathrm{tr}(S\Theta) - \log \det(\Theta) + \lambda \sum_{j \neq k} |\Theta_{jk}| \right)
$$

where S is the sample covariance, and λ is a regularization parameter.

Each entry of the precision matrix represents the conditional dependence between the two corresponding
variables, where 0 means that they are conditionally independent. Then this can be used to build a graph
based on the non-zero entries of the precision matrix, representing the variables that conditionally depend
on each other.

The higher the regularization parameter λ, the fewer features are selected, which causes the graph to become
sparser. This is a tradeoff: a denser graph has more information, but it increases model complexity
and makes it harder to learn; while a sparser graph can remove noise and focus on important features, but
it also risks removing important information.

### Model Architecture
We used a Graph Attention Network (GAT), which is a kind of Message-passing Graph Neural Network
(GNN), i.e. each node aggregates messages from its neighbors to learn node features [4, 7]. In particular,
the GAT employs an attention mechanism:

$$
x'_i = \alpha_{i,i}W_s x_i + \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} W x_j
$$

$$
\alpha_{i,j} = \frac{\exp\left(\mathrm{LeakyReLU}\left(a_s^\top W_s x_i + a^\top W x_j\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\left(\mathrm{LeakyReLU}\left(a_s^\top W_s x_i + a^\top W x_k\right)\right)}
$$
$$
These learnable attention weights allow the GAT to attend differently to different neighbors, which enables
more complex graph relationships to be captured.
Then, a sum pooling operation is done, i.e. the sum of all the node features is computed, so that the entire
graph’s features can be collected into a single tensor. Lastly, a linear layer (i.e. a fully connected layer) is
used to project the features into the output dimensions.
Xavier weight initialization is used, which draws initial values between a range with the goal of controlling
the variance of the initial weights, which can help mitigate vanishing/exploding gradients during training
[2].

We also employ several methods for regularization. Dropout layers are inserted between each GAT convolution layer, so that the model can learn robust features that don’t depend on particular nodes. We also use

residual skip connections between the layers, which can help smooth the loss landscape and help the model
learn simpler and stabler mappings, while also mitigating the vanishing gradient problem [3]. It has also
been demonstrated that skip connections improve training speed for GNNs [8].
Since this is a regression problem, mean-squared error (MSE) was selected as the loss function. The Adam
optimizer, which is a commonly used optimizer, was used to optimize on this.

### Baseline Model
We use a multi-layer perceptron (MLP) as a baseline, i.e. for comparison with our model. This is a simple
two-layer perceptron with ReLU activation. Dropout is also used between the layers for regularization. As
with our model, the loss function is MSE and the optimizer is Adam.

## Implementation
### Basic Environment and Equipment
Experiments were run on Google Colab with the Python 3 environment. To aid in handling complex
processes, an NVIDIA T4 GPU (15 GB RAM) was used.

### Data Preprocessing
To start the analysis, we begin by cleaning the available dataset. This is a manual process that involves
selecting the relevant sections in the publically available spreadsheet file, then organizing them into a csv
file. In addition to basic identifiers for SNPs, the cleaned dataset contains the type of changed nucleotide,
its location, and the Infection Type (BLUE-IT) and Severity (BLUE-SEV).
Then, the data is manipulated into a clean pandas dataframe containing all the SNPs and their mutation
types. In order for the model to properly train on the data, the SNP mutations are converted to categorical
data.
Data was then scaled by skpreprocess.StandardScaler(), which normalizes the data to zero mean and
unit variance. This is an important step that can help models train properly.
Then, for proper evaluation, we did a 60-20-20 train-validation-test split on the data. The validation dataset
is used for learning rate scheduling among other things, while the test dataset is used to actually evaluate
the model based on our chosen metrics.

<img width="1019" height="1033" alt="image" src="https://github.com/user-attachments/assets/2a4a1058-68f8-4f45-8fd6-215e5211d853" />
Figure 1: Graphical Lasso with λ = 0.3 

### Graphical Lasso
We did some hyperparameter tuning on the regularization parameter for the graphical lasso, so that we
would achieve a sparse but still meaningful graph. The goal was to allow the GNN further down the pipeline
to focus on the SNPs with stronger interactions, which would cut down on model complexity and help it
learn on important features, but also populate the graph enough such that there are still enough connections
for the model to learn.
With some trial and error, we decided on λ = 0.3 for lasso, which produced an appropriately sparse graph
that still had enough edges for the model to learn, as shown in Figure 1. Empirically this also produced
good results.

### Graph Attention Network
After hyperparameter tuning with trial and error, we decided on the following hyperparameters:
* **Hidden dimensions:** $128$
* **Number of heads in GAT attention:** $8$
* **Dropout probability:** $0.1$
* **Batch size:** $64$
* **Learning rate:** $6 \times 10^{-3}$
* **Number of epochs:** $200$
* **Adam hyperparameters:** $\beta_1 = 0.9, \beta_2 = 0.999$
* **Adam stability:** $\epsilon = 10^{-8}$
* **Weight decay:** $0$

In particular, note that weight decay is 0, possibly because dropout and skip connections already provide
enough regularization.
We also use the ReduceLROnPlateau scheduler with a factor of 0.5 and a patience of 10. Its purpose is to
reduce the learning rate during training when the validation loss doesn’t improve in a certain interval, to
help the model escape local minima during the training process [5].

### MLP (baseline)
We tried to keep things similar, for the sake of comparison. We used the same optimizer and scheduler, and
we ended up choosing the same hyperparameters except for the learning rate, which was 1e-3; and the number
of heads (because the MLP doesn’t have attention heads). All other hyperparameters were set to the same
value, which empirically produced the best results.


## Results
### Overview
Since this is a regression problem, we assessed model performance on a held-out test set through Mean
Squared Error (MSE) and Mean Absolute Error (MAE).

<img width="230" height="70" alt="image" src="https://github.com/user-attachments/assets/7edfc954-6a43-4b8f-b7be-499c784d6806" />
Table 1: Evaluation Results

The results in Table 1 show that the GAT outperforms the baseline MLP on every metric, suggesting that
our method is indeed effective in predicting disease resistance from plant genetic markers.

### Further Analysis

Taking a closer look at the training and validation losses in Figure 2a, we can see that the GAT is able to
adequately learn the data. Although the loss fluctuates significantly in the beginning, both training and
validation loss stabilize at a similar value, suggesting that our model does not overfit.

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5c2320b4-b87e-45e1-84e8-2a0a051d258a" />
(a) GAT

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/f2f9a2be-32ce-4477-a977-4ffb22371b7c" />
(b) MLP

Figure 2: Training and Validation Loss Curves


The large fluctuation in loss at the beginning can be attributed to the unfortunately small size of the data,
which leads to abrupt changes in the gradient during training. A larger dataset would help stabilize the
training and allow our model to learn better.

That said, it appears that the GAT model is able to overcome this fairly well, possibly due to the regu-
larization employed in the model, as well as the feature selection done through the graphical lasso—this

encourages the model to capture meaningful relationships between different features, which could help it
learn robust features that can generalize well.
However, the same cannot be said for the MLP, the baseline model. As seen in Figure 2b, the training loss
quickly drops in the first 25 epochs, while the validation loss barely changes at all. This suggests that the
model is overfitting on the training data, and it is failing to generalize to the whole dataset. While the GAT
learns on a sparse graph with implicit feature selection, the MLP has no such feature engineering built into
it, which means it is learning more complex features. This could explain the lack of generalization.

## Conclusion

To summarize, our project was a success. Our measure of success was to successfully create a working GNN
that outperforms the baseline MLP model, which we did, using MSE and MAE as metrics.
Although we mentioned in the proposal that this is a novel and complex project with no guarantee of success,
not only does our model significantly outperform the baseline model, but it also demonstrates desirable
properties during training; namely, it is able to learn and generalize on the data without overfitting, in
contrast to the MLP, which fails to learn.
However, owing to the unexpected personal circumstances of one of our members, we did not have the
manpower to complete our extension goal, which was to evaluate the model on another dataset. That said,
the core results of our model are successful, suggesting that GATs are able to predict disease resistance in
plants from SNP data. Perhaps applying our model architecture to other datasets could be an interesting
future direction to explore.
To conclude, in this project, we managed to explore a complex topic: using the graphical lasso to build a
graph from the dataset, and using a graph attention network to learn to predict plant disease resistance. This
is a highly relevant problem with significant real world implications, and we were able to make use of what
we had learned (data preprocessing, feature selection, regularization, neural networks), as well as outside
research, to accomplish the task. Ultimately, despite the challenges we faced, we were able to successfully
implement a model that outperforms a baseline MLP model.

# References
[1] Peter Bulli, Junli Zhang, Shiaoman Chao, Xianming Chen, and Michael Pumphrey. Genetic
architecture of resistance to stripe rust in a global winter wheat germplasm collection. G3
Genes—Genomes—Genetics, 6(8):2237–2253, 08 2016.

[2] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural net-
works. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International

Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Re-
search, pages 249–256, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR.

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recogni-
tion, 2015.

[4] PyG Team. conv.GATConv. https://pytorch-geometric.readthedocs.io/en/stable/generated/
torch_geometric.nn.conv.GATConv.html, 2024. Accessed: 2024-4-20.
[5] PyTorch Team. ReduceLROnPlateau. https://pytorch.org/docs/stable/generated/torch.optim.
lr_scheduler.ReduceLROnPlateau.html. Accessed: 2024-4-20.
[6] Shriprabha R. Upadhyaya, Monica F. Danilevicz, Aria Dolatabadian, Ting Xiang Neik, Fangning Zhang,

Hawlader A. Al-Mamun, Mohammed Bennamoun, Jacqueline Batley, and David Edwards. Genomics-
based plant disease resistance prediction using machine learning. Plant Pathology, 73(9):2298–2309, 2024.

[7] Petar Veliˇckovi ́c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua Bengio.
Graph Attention Networks, 2018.

[8] Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, and Kenji Kawaguchi. Optimization of Graph Neural Net-
works: Implicit Acceleration by Skip Connections and More Depth, 2021.

