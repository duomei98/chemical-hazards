# chemical-hazards

## Milestone 2: Preprocessing

Link to Jupiter Notebook: [https://colab.research.google.com/drive/156GEnqJ5HGwsO11rVrDdKndMWp0kH7Ew#scrollTo=tdiI1ycHs-IS](url)

Given the size of the dataset, data preprocessing first required splitting the data into multiple XML files. We then took several steps to clean and extract the necessary chemical hazard and SMILES data. We parsed the XML files to extract GHS hazard codes with their associated CID, which we saved in a TSV file. Similarly, we extracted SMILES data with associated CIDs, and saved it in a separate TSV file. 

To combine the datasets we first convert GHS codes to a one-hot encoding based on hazard class, then merged it with SMILES data based on CID, dropping the original GHS codes. This resulted in organized dataframe indicating a chemical's SMILES string and a binary hazard indicator for each hazard.
