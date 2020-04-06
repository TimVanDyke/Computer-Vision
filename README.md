# Computer-Vision

Create a "datasets" folder in the directory and extract these datasets in it. https://drive.google.com/open?id=1pmsn73Wp7-R1CRfJsY08la7HZCceS_8o

Then open train.py and change the dataset variable. You can also play around with the number of trainnings and number of hidden nodes in each layer to find the best combination. 

To create a nework run: ```python train.py <datasetName> <saveName> <iterations>```\
```datasetName``` = Name of the dataset i.e. mnist/letter/balanced/digits/byclass/bymerge\
```saveName``` = Nework will saved with this name in the networks folder\
```iteretion``` = number of iterations for the training\

To test a nework run: ```python test.py <datasetName> <saveName>```\
```datasetName``` = Name of the dataset i.e. mnist or letters\
```saveName``` = Name of the saved network in the networks folder without the extension\

