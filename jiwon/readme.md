# BPR
## Environment
---
### python package
Python package
You have to install the following packages before executing this code.
- python==3.6
- pytorch==1.3.1
- numpy==1.15.4
- pandas==0.23.4
You can install these package by executing the following command or through anaconda.
code 
### Reference
- https://github.com/sh0416/bpr
## Usage
---
### Preprocess
``` bash
python preprocess.py
```
### Training
- 데이터 이름은 'ml-1m.pickle'로 설정되어 있으며 preprocess에서 생성됨
``` bash
python train.py
```
### Inference
``` bash
python train.py --mode submission
```

# Checkyear.py
## Usage
---
- line 11에 item과 year 정보를 작성한 파일명 입력
- line 12에 user과 year 정보를 작성한 파일 입력
- line 38의 file1 = 뒤에 확인하고자 하는 파일명 입력
```'bash
python checkyear.py
```