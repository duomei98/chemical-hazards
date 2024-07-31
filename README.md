# INTRODUCTION
In brainstorming ideas for our project, we focused on topics that we found were both interesting and practical. We settled on the idea of using the structure of chemicals to predict what hazards were associated with it; it fulfilled these criteria, and datasets on this topic were readily available. Since a chemical's properties are largely tied to its molecular structure and the bonds between its constituent elements, we were curious to see if a model could successfully predict GHS (Globally Harmonized System) hazard class given a chemical's SMILES (Simplified Molecular Input Line Entry System) string, as well as its charge and type (if applicable). 

Through the course of our investigation, we discovered that our model’s ability to predict certain hazards is related not only to the chemical's properties but also the nature of the hazard. Analyzing these relationships was very fun and rewarding, and we believe with some key improvements, our model can be used in laboratory settings in safety protocol and chemical handling.

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
```
# Split training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Perform multilabel classification with Logistic Regression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model = MultiOutputClassifier(LogisticRegression(max_iter= 300000)).fit(X_train, y_train)
yhat_test = model.predict(X_test)
yhat_train = model.predict(X_train)
print(yhat_test)

print(f"Test Accuracy Score: {accuracy_score(y_test, yhat_test)}")
print(f"Train Accuracy Score: {accuracy_score(y_train, yhat_train)}")
print(classification_report(y_test, yhat_test))
```

Figure 1: Logistic regression framework for GHS Hazard classification
![Model 1 flowchart; ML](https://github.com/user-attachments/assets/86f2ac9c-2291-45cb-959d-a58f9370b5bc)

## Model 2
Notebook link: [model_final_(1).ipynb](url)

Our second model is a Convolutional Neural network. We preprocessed our input: first pruning observations with SMILES strings of length greater than 100, then augmenting our data, generating alternative SMILES strings for the hazard classes 'explosive', 'oxidizer', and 'pressurized' using the SMILES randomizer from the SMILES-enumeration library. 
```
#repeats is the number of times we attempt to add another smile
def addSmiles(hazards, repeats, shortdf):
    smiles = set()
    newdf = [shortdf]
    for hazard in hazards:
        print(hazard.upper(), ':\n')
        comps = shortdf[shortdf[hazard] == 1]
        smiles.update(comps['SMILES'])
        for comp in comps.iterrows():
            #print(comp[1][0])     #uncomment to see which smiles is ducking up the code
            try: #some SMILES are not compatible with the smiles randomizer, we'll ignore those
                alt = sme.randomize_smiles(comp[1][0])
            except:
                print(comp[1][0], "isn't quite valid")
                continue;
            for i in range(repeats):
                #print(np.transpose(np.array(comp[1].to_numpy())))
                row = pd.DataFrame(np.transpose(comp[1].to_numpy()[:,np.newaxis]), columns=comp[1].index)
                #return row
                #print(comp[1][0])
                alt = sme.randomize_smiles(comp[1][0])
                if alt not in smiles:
                    smiles.add(alt)
                    #print(alt)
                    row['SMILES']=alt
                    newdf.append(row)
    return pd.concat(newdf, axis=0)

shortdf = addSmiles(['explosive', 'oxidizer', 'pressurized'], 100, shortdf)
```

Next, we tokenized the remaining strings using the basic SMILES tokenizer from deepchem. We created a dictionary to classify ions based on their charge (neutral, positive, or negative) and type (metal or organic). Finally, we convert the tokenized input into a matrix of one-hot encoded vectors that we can feed into a neural network. 
```
type2idx = {  # key:
    'nm': 0,  # neutral metal
    '+m': 1,  # positively charged metal
    '-m': 2,  # negatively charged metal
    'no': 3,  # neutral organic
    '+o': 4,  # positively charged organic
    '-o': 5   # negatively charged organic
}
```

We first built a CNN model consisting of a ConvID layer with 32 filters, a max pooling layer, and three dense layers. Then, we tuned the model using Keras hyperparameter tuning, adjusting parameters such as number of hidden layers, activation function, number of units, and learning rate. 
```
# tuning

def buildHPmodel(hp):
    activation= hp.Choice("activation", ["sigmoid", "tanh",'relu'])
    finalactivation= hp.Choice("output activation", ["sigmoid", "tanh"])
    units =  int(hp.Int("units", min_value=16, max_value=64, step=2, sampling='log'))
    
    cnn = Sequential();
    cnn.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(100,22)))
    cnn.add(MaxPooling1D())
    cnn.add(Flatten())

    for i in range(hp.Int("num_layers", 2, 4)):
        cnn.add(Dense(units=units, activation=activation))

    cnn.add(Dense(units = 9, activation = finalactivation))
    opt = keras.optimizers.RMSprop(learning_rate = 0.05)
    cnn.compile(optimizer = opt, loss = 'mean_squared_error', metrics=['accuracy'])
    return cnn
```
Next, we further evaluated the best five models with a greater number of epochs. 

Based on the best performing hyperparameters, we defined our final CNN model and evaluated it on metrics such as accuracy, precision, and recall.  

# RESULTS
## Pre-Processing
No results applicable

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

Figure 4: Confusion Matrix for all Classes
![Model 1 Framework (4)](https://github.com/user-attachments/assets/f52f58cb-c440-415b-8222-fc6bc10cdf22)

Figure 5: Recall and Precision Bar Charts
![Model 1 precision recall numbers](https://github.com/user-attachments/assets/89f17440-2b2c-4c06-a3ed-8acb54837163)

## Model 2 (Final Model) 
Figure 6: Confusion Matrix for all Classes
![Model 1 Framework (8)](https://github.com/user-attachments/assets/3c305b24-c00a-4322-a6ff-e1416263c110)

Figure 7: Accuracy Bar Chart
![Model 1 Framework (10)](https://github.com/user-attachments/assets/5555f792-bf7c-4dab-a5a8-4fa007b894f5)

Figure 8: Precision Bar Chart
![Model 1 Framework (11)](https://github.com/user-attachments/assets/a6592b0a-d78e-4453-91ed-f37e40cfb3ef)

Figure 9: Recall Bar Chart
![Model 1 Framework (9)](https://github.com/user-attachments/assets/1f721f7b-3c69-49e0-90c8-519ee06d793e)

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

When pre-processing our data for a neural network, we decided to remove all SMILES strings with length greater than 100, which represented around 3% of our total data, for the sake of simplifying our model. Since these strings were outliers in our dataset, removing them should not significantly affect the predictive ability of our model.

Given the issues we encountered in Model 1 with imbalanced classes, we needed to balance our data in some way. We tackled this issue by augmenting our data using SmilesEnumerator, which generates all possible SMILES forms of a molecule. For the classes in which we had the fewest observations (explosive, oxidizer, and pressurized), we added alternate SMILES into our dataset to balance our classes. These strings are alternate forms of compounds present in those classes. 

We first attempted a basic neural network structure, but it was too simple for the structural complexity of our dataset, leading to underfitting. Thus, we turned our attention to CNNs, which works well with sequential and spatial data, such as the molecular structure represented in SMILES strings. 

After building a CNN with 32 filters and three Dense hidden layers, we decided to hypertune our model. Our top 5 models all had a validation accuracy score of around 65%, with models with more layers and units yielding better results in error on both test and training data. Furthermore, we found that having all activation functions as sigmoid yielded the best results. This is because our problem is a multi-label classification problem, with independent probabilities per class.

Using these insights, we built our final model, and retrained it with a larger epoch size of 50 to evaluate how it performs, finding an overall accuracy of 65% on test data and around 70% for training data. This signals to us that our model is underfitting, and could benefit from further hypertuning.

Since the cost of false negatives (failing to report a hazard) is high, our target metric was recall. Across all 9 labels, explosive, oxidizer, and irritant had the highest recall, of around 94%, with flammable and pressurized performing decently well at around 70-80%. We had the most robust data for irritants, and our model appeared to train well on our augmented SMILES strings in the clases explosive, oxidizer, and pressurized. However, the recall for corrosive, toxic, health hazard, and environmental hazard were all below 50%. 

For health and environmental hazards, these results are reasonable. It should be rather difficult to predict health / environmental hazards as this information is not explicitly stored in the chemical itself but rather in the interactions between the chemical and other biomolecules. However, we are not sure why corrosive and toxic compounds performed poorly, since we had a decently robust set of training data for these classes.

In conclusion, our model predicts 5 out of the 9 classes well but is far from perfect. We should've validated our data further using cross validation. Another improvement might be to build one model per hazard, which might lead to better performance by allowing each model to specialize in detecting specific hazards. Finally, further data augmentation and hyperparameter tuning could improve the recall for the poorly performing classes.

# CONCLUSION

We think the idea of using molecular structure in the form of SMILES strings to predict chemical hazards is really cool, and we're glad we chose to explore this topic. Though the process of building our models was challenging, we greatly enjoyed assessing the relationships between hazards and evaluating our models' predictive performance. 

For instance, our model struggled to predict health and environmental hazards, but this makes sense since these hazards are more related to chemical handling and storage rather than any intrinsic property of the compound. However, it is concerning that our model does not predict corrosives well, despite the fact that they are the second most common hazard on PubChem. This suggest that corrosives are probably common in practical settings, and thus it is very important to correctly identify this hazard.

If we were to redo our project, we would probably try to augment the underperforming classes further, which turned out to be our main obstacle. We would also avoid wasting time with a Logistic Regression model, which performed very poorly, and used that time to perfect our neural network. Given more time, we would like to incorporate cross-validation and further hyperparameter tuning to improve our model's validity and performance.

We are excited about possibly improving our model in the future since there are many different avenues we can take. We can try building separate CNNs per hazard, trying out NLP strategies on our data, or using transformer models or other advanced architectures. Our group had a lot of fun working together on this project, and while our final model is far from perfect, we learned a lot through the process.

# STATEMENT OF COLLABORATION
Anna He. Contributed ideas and input during team meetings. Coded most of the data exploration portion. Wrote the Milestone 2 submission and wrote the final write-up

Yuehua Xie: coded most of the data preprocessing + augmentation portion + our final model. Wrote the Milestone 3 submission.
