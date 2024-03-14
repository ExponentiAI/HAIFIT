# HAIFIT

***
HAIFIT: Human-Centered AI for Fashion Image Translation. [**Arxiv**](https://arxiv.org/abs/2403.08651)

Our model weights is avialable [**checkpoint**](https://drive.google.com/drive/folders/1DW2O9xIiL_wb4BDz06PflUqSq_n9v-Lf?usp=drive_link)

## 1. Data preparation
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


## 2. Train and Test
### 2.1 Train
set **model=1** in **main.py** and run:
```yaml
python main.py
```

### 2.2 test
set **model=2** in **main.py** and run:
```yaml
python main.py
```
Note: you should change your checkpoint path.

## 3. Reference
If you find our code or dataset is useful for your research, please cite us, Thanks.
```yaml
@article{jiang2024haifit,
  title={HAIFIT: Human-Centered AI for Fashion Image Translation},
  author={Jiang, Jianan and Li, Xinglin and Yu, Weiren and Wu, Di},
  journal={arXiv preprint arXiv:2403.08651},
  year={2024}
}
```

Or

```yaml
Jia Jiang, Xing Li, Weiren Yu, et al. HAIFIT: Human-Centered AI for Fashion Image Translation.[J]. arXiv preprint arXiv:2403.08651, 2024.
```
