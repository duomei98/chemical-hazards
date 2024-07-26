# chemical-hazards

## Milestone 2: Preprocessing

Link to Jupyter Notebook: [https://colab.research.google.com/drive/156GEnqJ5HGwsO11rVrDdKndMWp0kH7Ew#scrollTo=tdiI1ycHs-IS](url)

Given the size of the dataset, data preprocessing first required splitting the data into multiple XML files. We then took several steps to clean and extract the necessary chemical hazard and SMILES data. We parsed the XML files to extract GHS hazard codes with their associated CID, which we saved in a TSV file. Similarly, we extracted SMILES data with associated CIDs, and saved it in a separate TSV file. 

To combine the datasets we first convert GHS codes to a one-hot encoding based on hazard class, then merged it with SMILES data based on CID, dropping the original GHS codes. This resulted in organized dataframe indicating a chemical's SMILES string and a binary hazard indicator for each hazard.

## Milestone 3: Preprocessing + Our first model!

Our first model is a multilabel logistic regression model (see models.ipnyb). We additionally further preprocessed our input by tokenizing the SMILES strings using the regex-based basic SMILES tokenizer from deepchem.

As the goal of our model is to assign labels (chemical hazards) to various SMILES strings, our metric of choice was recall, since reporting an extraneous hazard is less of an issue than not reporting the applicable hazards. The recall of our model varies wildly across the different labels, from a recall of 3% for explosives to a recall of 99% for irritants, with an overall recall of 72% on the test data. The accuracy of the model was 60.3% for the test data and 59.9% on the training data, indicating that the model likely lies in the lefthand side (underfitting) section of the fitting graph. 

Our conclusion is that our model does not work very well. With the exception of irritants, the recall for all other chemical hazards sits well below 50% (27% for oxidizers, 11% for flammables, and in the single digits for all other hazards). Because a SMILES string contains primarily structural data that was not utilized properly in this model, we intend to try using a neural network for our next model.
