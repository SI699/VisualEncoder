#VisualEncoder
This projects aims to train a auto-encoder to grab the visual features of Chinese characters (Hanzi).
## Usage
### Generate dataset
You can follow the instruction in ```data_process.``` to generate the dataset.

### Train the model
You can change the parameters in ```config/config.yml``` or create your own config file. Then run ```python train.py --config config/config.yml```

### Visualize the result
Follow the instruction in ```visualize.ipynb``` to visualize the importance of different parts of input image.