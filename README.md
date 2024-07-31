# INTRODUCTION
In brainstorming ideas for our project, we focused on topics that we found were both interesting and practical. We settled on the idea of using the structure of chemicals to predict what hazards were associated with it; it fulfilled these criteria, and datasets on this topic were readily available. Since a chemical's properties are largely tied to its molecular structure and the bonds between its constituent elements, we were curious to see if a model could successfully predict GHS (Globally Harmonized System) hazard class given a chemical's SMILES (Simplified Molecular Input Line Entry System) string, as well as its charge and type (if applicable). 

Through the course of our investigation, we discovered that our model’s ability to predict certain hazards is related not only to the chemical's properties but also the nature of the hazard. Analyzing these relationships was very fun and rewarding, and we believe our model can be used in laboratory settings in safety protocol and chemical handling.

# METHODS
## Pre-Processing
Notebook link: [milestones.ipynb](url)

Data preprocessing first required splitting the data into multiple XML files. We then took several steps to clean and extract the necessary chemical hazard and SMILES data. We parsed the XML files to extract GHS hazard class with their associated CID, which we saved in a TSV file. Similarly, we extracted SMILES data with associated CIDs, and saved it in a separate TSV file. 

To combine the datasets we first converted GHS classes to a one-hot encoding based on hazard class, then merged it with SMILES data based on CID, dropping the original GHS classes. This resulted in an organized data-frame indicating a chemical's SMILES string and a binary hazard indicator for each hazard.

## Data Exploration
Notebook link: [milestones.ipynb](url)

When exploring our data, we first checked our data for null values, then analyzed the count distribution of our hazard class features. Next, we investigated the correlation between the hazard classes using a heatmap. Finally, we enhanced our data by adding a SMILEs string length attribute. 

## Model 1
Notebook link: [models.ipynb](url)

Our first model is a multilabel logistic regression model. We additionally further preprocessed our input by tokenizing the SMILES strings using the regex-based basic SMILES tokenizer from deepchem. Then, we evaluated our data for each class, focusing on the recall metric. 

Figure 1: Logistic regression framework for GHS Hazard classification
![Model 1 flowchart; ML](https://github.com/user-attachments/assets/86f2ac9c-2291-45cb-959d-a58f9370b5bc)

## Model 2
Notebook link: [model_final_(1).ipynb](url)

Our second model is a Convolutional Neural network. We preprocessed our input: first pruning observations with SMILES strings of length greater than 100, then augmenting our data, generating alternative SMILES strings for the hazard classes 'explosive', 'oxidizer', and 'pressurized' using the SMILES randomizer from the SMILES-enumeration library. 

Next, we tokenized the remaining strings using the basic SMILES tokenizer from deepchem. We created a dictionary to classify ions based on their charge (neutral, positive, or negative) and type (metal or organic). Finally, we convert the tokenized input into a matrix of one-hot encoded vectors that we can feed into a neural network. 

We first built a CNN model consisting of a ConvID layer with 32 filters, a max pooling layer, and three dense layers. Then, we tuned the model using Keras hyperparameter tuning, adjusting parameters such as number of hidden layers, activation function, number of units, and learning rate. Next, we further evaluated the best five models with a greater number of epochs. 

Based on the chosen hyperparameters, we defined our final CNN model and evaluated it on metrics such as accuracy, precision, and recall.  

# RESULTS
## Data Exploration
Figure 2: Count distribution of class attributes: GHS Hazard Classes
![GHS Hazard counts; bargraph](https://github.com/user-attachments/assets/39bff22f-823f-457c-ac86-0e8cd669318e)

Figure 3: Correlational Heatmap of GHS Hazard Classes
![heatmap](https://github.com/user-attachments/assets/778b5f0f-a3f3-4516-a9db-7e19b74770b3)

## Model 1
 
Classification Report:
| feature | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| explosive | 0.00 | 0.00 | 0.00 | 37 | 
| flammable | 0.49 | 0.11 | 0.18 | 2593 |
| oxidizer | 0.89 | 0.24 | 0.38 | 71 |
| pressurized | 1.00 | 0.03 | 0.06 | 60 |
| corrosive | 0.42 | 0.03 | 0.06 | 6910 |
| toxic | 0.66 | 0.04 | 0.08 | 3159 |
| irritant | 0.85 | 0.99 | 0.91 | 37490 |
| health hazard | 0.73 | 0.05 | 0.09 | 3294 |
| environmental hazard | 0.66 | 0.05 | 0.09 | 3640 |
| --- | --- | --- | --- | --- |
| micro average | 0.84 | 0.67 | 0.74 | 57254 | 
| macro average | 0.63 | 0.17 | 0.21 | 57254 | 
| weighted average | 0.76 | 0.67 | 0.63 | 57254 | 
| samples average | 0.84 | 0.72 | 0.75 | 57254 | 

Figure 4: Confusion Matrix 
![Model 1 Framework (4)](https://github.com/user-attachments/assets/f52f58cb-c440-415b-8222-fc6bc10cdf22)

Figure 5: Recall and Precision Bar Charts
![Model 1 precision recall numbers](https://github.com/user-attachments/assets/89f17440-2b2c-4c06-a3ed-8acb54837163)

## Model 2 (Final Model) (TO DO)
Confusion matrix, recall precision bar graph

# DISCUSSION
## Pre-Processing
Given the incredibly large size of database we were drawing from, we needed to carefully prune our data. First, we discarded all non-hazardous compounds, which were irrelevant to our goal. For the remaining compounds, the most important features to extract were CID (Compound Identifier), SMILES string, and associated GHS hazard codes. Since there are around 70 GHS hazard codes listed on PubChem, we decided to sort them into their hazard class, of which there are 9 in total. This approach avoids overcomplicating our model while still retaining the important information. 

For the same reason, we discarded all text and image-based data. We considered including numerical attributes such as boiling or melting point, but ultimately decided against it to narrow the scope of our project. Including more attributes might improve the predictive ability of our model, so there is area for improvement here. However, given the scale of our dataset, we opted to prioritize simplicity and speed. 

## Data Exploration
After pre-processing our data into a dataframe with SMILES string and one-hot encoded GHS Hazard classes as features, we wanted to examine our data for patterns. First, we looked at the counts distribution for each Hazard class and found that irritants made up about 65% of all data, while the classes 'explosive', 'oxidizer', and 'pressurized' summed to less than 1%. This was not unexpected, given that irritants are much more common than explosives, oxidizers, and pressurized compounds. 

However, this class imbalance was very problematic when it came time to build our models. In hindsight, it might've been smarter to deal with this challenge at this stage of our project. We could've oversampled the minority classes or randomly removed some of the irritant samples so that our data was more balanced.

Finally, we generated a correlation heatmap for the GHS Hazard classes. It showed low correlation between all class features, meaning that hazards are largely independent of each other. Based on this correlation, we determined that we did not need to prune any features.

## Model 1
While we suspected that a neural network might be the most effective model given the nature of our input, we wanted to test a variety of different models. Since our class features are binary outcomes, we chose logistic regression for our first model. 

As the goal of our model is to assign labels (chemical hazards) to various SMILES strings, our metric of choice was recall. This is because reporting an extraneous hazard is less of an issue than not reporting the applicable hazards. The recall of our model varies wildly across the different labels, from a recall of 3% for explosives to a recall of 99% for irritants, with an overall recall of 72% on the test data. The accuracy of the model was 60.3% for the test data and 59.9% on the training data, indicating that the model likely lies in the lefthand side (underfitting) section of the fitting graph. 

Our conclusion is that our model does not work very well. With the exception of irritants, the recall for all other chemical hazards sits well below 50% (24% for oxidizers, 11% for flammables, and in the single digits for all other hazards). Because a SMILES string contains primarily structural data that was not utilized properly in this model, we intended to try using a neural network for our next model.

## Model 2 

When pre-processing our data for a neural network, we decided to remove all SMILES strings with length greater than 100, which represented around 3% of our total data, for the sake of simplifying our model. Since these strings were outliers in our dataset, removing them should not affect the predictive ability of our model.

Then, given the issues we encountered in Model 1 with imbalanced classes, we needed to balance our data in some way. We tackled this issue by augmenting our data using SmilesEnumerator, which generates all possible SMILES forms of a molecule. For the classes in which we had the fewest observations (explosive, oxidizer, and pressurized), we added alternate SMILES into our dataset to balance our classes. (why does this method work?)

We first attempted a basic neural network structure, but it was too simple for the structural complexity of our dataset, with our model lying squarely in the underfitting portion of the fitting. Thus, we turned our attention to CNNs, which works well with sequential and spatial data, such as the molecular structure represented in SMILES strings. 

Again, our target metric was recall. We found that across all 9 labels, (insert some data summary similar to what we have in model 1). Our model appears to lie ___ in the fitting graph.

Then, we decided to hypertune our model and see how we can improve our model. Our top 5 models 

# CONCLUSION (TODO)
- Opinions:
  - What went well and what didn't, etc
  - Maybe a discussion of the dataset itself?
  - it should be rather difficult to predict hazards (especially health/environmental hazards) as this information is not explicitly stored in the chemical itself but rather in the interactions between the chemical and other biomolecules
- Future directions:
  - Building one model per hazard, trying out NLP strategies on the data, etc

# STATEMENT OF COLLABORATION (TODO)
