# PVNet

Python/PyTorch source code for the paper:

	A. Genovese, V. Bernardoni, V. Piuri, F. Scotti and F. Tessore, 
	"Photovoltaic Energy Prediction for New-Generation Cells with Limited Data: A Transfer Learning Approach," 
	2022 IEEE Int. Instrumentation and Measurement Technology Confe. (I2MTC), 2022, pp. 1-6, 
	doi: 10.1109/I2MTC48687.2022.9806492.
	
Project page:

[https://iebil.di.unimi.it/PVNet/index.htm](https://iebil.di.unimi.it/PVNet/index.htm)
    
Outline:
![Outline](https://iebil.di.unimi.it/PVNet/imgs/outline.jpg "Outline")

Citation:

	@InProceedings {i2mtc22,
    author = {A. Genovese and V. Bernardoni and V. Piuri and F. Scotti and F. Tessore},
    booktitle = {Proc. of the 2022 IEEE Int. Instrumentation and Measurement Technology Conf. (I2MTC 2022)},
    title = {Photovoltaic energy prediction for new-generation cells with limited data: A transfer learning approach},
    address = {Ottawa, ON, Canada},
    pages = {1-6},
    month = {May},
    day = {16-19},
    year = {2022},}

Main files:

- 1. Create_DB_pvlib/launch_create_db_pvlib.py: creation of database;
- 2. Pvlib_extract_features_v2_MLP/launch_pvlib_extract_features.py: feature extraction;
- 3. Pvlib_train_finetune_v3_MLP/launch_pvlib_train_finetune.py: photovoltaic energy prediction using neural networks and transfer learning.

Instructions:

1) cd to "1. Create_DB_pvlib" and run "launch_create_db_pvlib.py" to create the database
    
2) cd to "2. Pvlib_extract_features_v2_MLP" and run "launch_pvlib_extract_features.py" to extract the features
    
3) cd to "3. Pvlib_train_finetune_v3_MLP" and run "launch_pvlib_train_finetune.py" to train the neural networks for photovoltaic energy prediction.

Required packages:
In our configuration, we used a PVLib environment to run the "1. Create_DB_pvlib", while we used a PyTorch environment to run the "2. Pvlib_extract_features_v2_MLP" and the "3. Pvlib_train_finetune_v3_MLP". For the installed packages in each environment, see "packages_pvlib.txt" and "packages_pytorch.txt"

- PVLib: https://pvlib-python.readthedocs.io/en/stable/

- PyTorch: https://pytorch.org/

