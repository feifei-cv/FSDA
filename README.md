
## Install
1. Download the dataset from the [SSTDA](https://github.com/cmhungsteve/SSTDA) repository, [Dataset Link Here](https://www.dropbox.com/s/kc1oyz79rr2znmh/Datasets.zip?dl=0)
2. Unzip the zip file, and re-name the './Datasets/action-segmentation' folder as "./dataset"
3. Clone git repositories for this repo and Run the script for ASRF
```
git clone https://github.com/feifei-cv/FSDA.git
./scripts/install_asrf.sh
```

## Train 
1. train backbones first.
```
python train_backbone_mstcn.py 
python train_backbone_ASFormer.py 
```

2. train our model FSDA
```
python train_FSDA_mstcn.py
python train_FSDA_ASFormer.py 
```   

## Acknowledgements
We hugely appreciate for previous researchers in this field. Especially [MS-TCN](https://github.com/yabufarha/ms-tcn), [ASFormer](https://github.com/ChinaYi/ASFormer), [ASRF](https://github.com/yiskw713/asrf),
[HASR](https://github.com/cotton-ahn/HASR_iccv2021), made a huge contribution for future researchers like us!
