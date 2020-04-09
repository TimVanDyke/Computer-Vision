# Computer-Vision

Create a "datasets" folder in the directory and extract these datasets in it. https://drive.google.com/open?id=1pmsn73Wp7-R1CRfJsY08la7HZCceS_8o

Then open train.py and change the ```learning_rate``` and in nn.py change the ```p``` variables to use different dropout rates and prevent overfitting. You can also play around with the number of trainnings and number of hidden nodes in each layer to find the best combination. 

To create a network run: ```python train.py <dataset_name> <save_name> <iterations>```\
```dataset_name``` = Name of the dataset i.e. mnist/letter/balanced/digits/byclass/bymerge\
```save_name``` = Nework will saved with this name in the networks folder\
```iteration``` = number of iterations for the training

To test a network run: ```python test.py <dataset_name> <save_name>```\
```dataset_name``` = Name of the dataset i.e. mnist or letters\
```save_name``` = Name of the saved network in the networks folder without the extension

To use your own data run: ```python pic_to_letter <folder_location>```\
```folder_location``` = Folder that includes pictures of letters in png format i.e. ```C:\Users\user\Documents\GitHub\Computer-Vision\examle_letters```\
Example letters can be found in the ```example_letters``` folder.\
You can also change the network used for recognition in the ```pic_to_letter.py``` file.

To extract individual characters from an image and save as their own file: ```python find_letter.py <image_name>```\
```image_name``` = Name of the image file. Must be in the ```pictures``` directory.\
Example: ``` python find_letter.py test.png ```