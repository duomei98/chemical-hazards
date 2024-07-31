# INTRODUCTION
In brainstorming ideas for our project, we focused on topics that we found were both interesting and practical. We settled on the idea of using the structure of chemicals to predict what hazards were associated with it; it fulfilled these criteria, and datasets on this topic were readily available. Since a chemical's properties are largely tied to its molecular structure and the bonds between its constituent elements, we were curious to see if a model could successfully predict GHS (Globally Harmonized System) hazard class given a chemical's SMILES (Simplified Molecular Input Line Entry System) string, as well as its charge and type (if applicable). 

Through the course of our investigation, we discovered that our model’s ability to predict certain hazards is related not only to the chemical's properties but also the nature of the hazard. Analyzing these relationships was very fun and rewarding, and we believe our model can be used in laboratory settings in safety protocol and chemical handling.

# METHODS
## Pre-Processing
Data preprocessing first required splitting the data into multiple XML files. We then took several steps to clean and extract the necessary chemical hazard and SMILES data. We parsed the XML files to extract GHS hazard class with their associated CID, which we saved in a TSV file. Similarly, we extracted SMILES data with associated CIDs, and saved it in a separate TSV file. 

To combine the datasets we first converted GHS classes to a one-hot encoding based on hazard class, then merged it with SMILES data based on CID, dropping the original GHS classes. This resulted in an organized data-frame indicating a chemical's SMILES string and a binary hazard indicator for each hazard.

## Data Exploration
When exploring our data, we first checked our data for null values, then analyzed the count distribution of our hazard class features. Next, we investigated the correlation between the hazard classes using a heatmap. Finally, we enhanced our data by adding a SMILEs string length attribute. 

## Model 1
Our first model is a multilabel logistic regression model. We additionally further preprocessed our input by tokenizing the SMILES strings using the regex-based basic SMILES tokenizer from deepchem. Then, we evaluated our data for each class, focusing on the recall metric. 

Figure 1: Logistic regression framework for GHS Hazard classification (I included this bc it seems like a lot of logistic regression model-based studies have a flowchart for some reason, delete maybe)
![Model 1 flowchart; ML](https://github.com/user-attachments/assets/86f2ac9c-2291-45cb-959d-a58f9370b5bc)

## Model 2 (Final Model) (TODO)
Our second model is a Convolutional Neural network. Again, we preprocessed our input: first pruning observations with SMILEs strings of length greater than 100, then tokenizing the remaining strings using the basic SMILEs tokenizer from deepchem. Then, we created a dictionary to classify ions based on their charge (neutral, positive, or negative) and type (metal or organic). Finally, we convert the tokenized input into a 22-dimensional one-hot encoded vector that we can feed into a neural network. 

Our CNN model consists of __ hidden layers and …. A short description (idk if this part is necessary but well)

# RESULTS
## Data Exploration
Figure 2: Count distribution of class attributes: GHS Hazard Classes
![GHS Hazard counts; bargraph](https://github.com/user-attachments/assets/39bff22f-823f-457c-ac86-0e8cd669318e)

Figure 3: Correlational Heatmap of GHS Hazard Classes
![heatmap](https://github.com/user-attachments/assets/778b5f0f-a3f3-4516-a9db-7e19b74770b3)

## Model 1 (Edit maybe, i'm not really sure which figures are best)

Figure 4: Confusion Matrices per class
![model 1 confusion matrices](https://github.com/user-attachments/assets/95ea4d9f-c59b-42c8-9894-e9d78ea78e1a)

Figure 5: ROC Curve per class
![Model 1 ROC curve](https://github.com/user-attachments/assets/250268fc-cea8-428a-9cc3-d8eeba08ae9f)

Figure 6: Recall and Precision Bar Charts
![Model 1 precision recall barcharts](https://github.com/user-attachments/assets/a0930717-889f-45a7-ab84-2f76336e0350)

## Model 2 (Final Model) (TO DO)
wahoo

# DISCUSSION
## Pre-Processing
Given the incredibly large size of database we were drawing from, we needed to carefully prune our data. First, we discarded all non-hazardous compounds, which were irrelevant to our goal. For the remaining compounds, the most important features to extract were CID (Compound Identifier), SMILES string, and associated GHS hazard codes. Since there are around 70 GHS hazard codes listed on PubChem, we decided to sort them into their hazard class, of which there are 9 in total. This approach avoids overcomplicating our model while still retaining the important information. 

For the same reason, we discarded all text and image-based data. We considered including numerical attributes such as boiling or melting point, but ultimately decided against it to narrow the scope of our project. Including more attributes might improve the predictive ability of our model, so there is area for improvement here. However, given the scale of our dataset, we opted to prioritize simplicity and speed. 

## Data Exploration
After pre-processing our data into a dataframe with SMILES string and one-hot encoded GHS Hazard classes as features, we wanted to examine our data for patterns. First, we looked at the counts distribution for each Hazard class and found that irritants made up about 65% of all data, while the classes 'explosive', 'oxidizer', and 'pressurized' summed to less than 1%. This was not unexpected, given that irritants are much more common than explosives, oxidizers, and pressurized compounds. 

However, this class imbalance was very problematic when it came time to build our models. In hindsight, it might've been smarter to deal with this challenge at this stage of our project. We could've oversampled the minority classes or randomly removed some of the irritant samples so that our data was more balanced.

Finally, we generated a correlation heatmap for the GHS Hazard classes. It showed low correlation between all class features, meaning that hazards are largely independent of each other. Based on this correlation, we determined that we did not need to prune any features.

## Model 1
While we suspected that a neural network might be the most effective model given the nature of our input, we wanted to test a variety of different models. Since that our class features are binary outcomes, we chose logistic regression for our first model. 

As the goal of our model is to assign labels (chemical hazards) to various SMILES strings, our metric of choice was recall. This is because reporting an extraneous hazard is less of an issue than not reporting the applicable hazards. The recall of our model varies wildly across the different labels, from a recall of 3% for explosives to a recall of 99% for irritants, with an overall recall of 72% on the test data. The accuracy of the model was 60.3% for the test data and 59.9% on the training data, indicating that the model likely lies in the lefthand side (underfitting) section of the fitting graph. 

Our conclusion is that our model does not work very well. With the exception of irritants, the recall for all other chemical hazards sits well below 50% (27% for oxidizers, 11% for flammables, and in the single digits for all other hazards). Because a SMILES string contains primarily structural data that was not utilized properly in this model, we intend to try using a neural network for our next model.

## Model 2 (TODO)

# CONCLUSION (TODO)
- Opinions:
  - What went well and what didn't, etc
  - Maybe a discussion of the dataset itself?
  - it should be rather difficult to predict hazards (especially health/environmental hazards) as this information is not explicitly stored in the chemical itself but rather in the interactions between the chemical and other biomolecules
- Future directions:
  - Building one model per hazard, trying out NLP strategies on the data, etc

# STATEMENT OF COLLABORATION (TODO)
