# HAIFIT

***
HAIFIT: Human-Centered AI for Fashion Image Translation.

Our model weights is avialable [**checkpoint**](https://drive.google.com/drive/folders/1DW2O9xIiL_wb4BDz06PflUqSq_n9v-Lf?usp=drive_link)

## Data preparation
For datasets that have paired sketch-image data, the path should be formatted as:
```yaml
./dataset/trainA/  # training reference (sketch)
./dataset/trainB/  # training ground truth (image)
./dataset/testA/  # testing reference (sketch)
./dataset/testB/  # testing ground truth (image)
```
After that, the configuration should be specified in config file in:
```yaml
./config.yml  # config file
```
Our Sketch-to-Image synthesis dataset is avialable [**HAIFashion**](https://drive.google.com/file/d/18nQfq7I7XUwXVFOqNbKmiyOnWaBJmw-_/view?usp=drive_link).


## Train and Test
### Train
set **model=1** in **main.py** and run:
```yaml
python main.py
```

### test
set **model=2** in **main.py** and run:
```yaml
python main.py
```
Note: you should change your checkpoint path.
