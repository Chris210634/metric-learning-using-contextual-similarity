# Supervised Metric Learning for Retrieval via Contextual Similarity Optimization

![](https://github.com/Chris210634/metric-learning-using-contextual-similarity/raw/main/figures/intuition.png)

![](https://github.com/Chris210634/metric-learning-using-contextual-similarity/raw/main/figures/rebuttal1.png)

## Downloading Data
Run `download_data.sh` to download all four datasets. Note that there are two places you need to edit before running this script: (1) replace `<data_dir>` with the location where you want the data to live, generally speaking you want fast I/O to this location, e.g. scratch disk; (2) replace `<git_dir>` with the location of your local git clone (this directory). SOP, CUB and Cars are fast; iNaturalist will take 2-4 hours to download and process. For In-Shop, you must download the images manually from the authors' link (img.zip not img_highres.zip)

We provide code for both 224 x 224 and 256 x 256 results.

## Running our Algorithm (224 x 224 image resolution)
Our 224x224 setup follows the Proxy-Anchor implementation on GitHub: [link](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020). They are located in the `224x224` folder.

### CUB
```
cd 224x224
python train.py --loss Hybrid --dataset cub --batch-size 128 --eps 0.04 --lam 0.05 \
--lr 0.00014 --warm 0 --bn-freeze 1 --gamma 0.1 --testfreq 10 --lr-decay-step 10 \
--epochs 60 --lr-decay-gamma 0.3 --xform_scale 0.16 --bottleneck linear --data_root <data_dir>/CUB_200_2011
```
### Cars
```
cd 224x224
python train.py --loss Hybrid --dataset cars --batch-size 64 --eps 0.05 --lam 0.2 \
--lr 8e-05 --warm 0 --bn-freeze 1 --gamma 0.1 --testfreq 10 --lr-decay-step 20 \
--epochs 160 --lr-decay-gamma 0.5 --xform_scale 0.12 --bottleneck linear --data_root <data_dir>/car196
```
### SOP
```
cd 224x224
python train.py --loss Hybrid --dataset SOP --batch-size 128 --eps 0.05 --lam 0.2 \
--lr 6e-05 --warm 0 --bn-freeze 1 --gamma 0.1 --testfreq 5 --lr-decay-step 15 \
--epochs 80 --lr-decay-gamma 0.3 --xform_scale 0.08 --bottleneck identity --hierarchical 1 --data_root <data_dir>/Stanford_Online_Products
```
### Baselines
We carefully tuned our baselines for results in Tables 2 and 3. The optimal hyperparameters for each baseline is different. They are listed in the tables below:

| CUB | learning rate | warm start epochs | samples per class | gamma | bottleneck |
| --loss | --lr | --warm | --IPC | --gamma | --bottleneck |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Contrastive  | 2e-5 | 'linear' | 0.0 | 4 | 0 | 
| Roadmap | 2e-5 | 'linear' | 0.1 | 4 | 0 | 
| Triplet' | 8e-5 | 'identity' | 0.1 | 4 | 0 | 
| MS+miner' | 0.00016 | 'linear' | 0.0 | 4 | 0 | 
| Proxy_Anchor' | 0.00016 | 'linear' | 0.0 |  0 |  10 | 
| Proxy_NCA' | 0.00016 | 'linear' |  0.0 |  0 |  10 | 
| Hybrid (Contextual) | 0.00014 | 'linear' |  0.1 |  4 |  0 | 

## Running our Algorithm (256 x 256 image resolution)
`main.py` is the main python script containing our implementation. Here is how you run it to reproduce our results:

### CUB
```python main.py --dataset CUB --loss hybrid --lr 8e-05 --root <data_dir>/CUB_200_2011```
### Cars
```python main.py --dataset Cars --loss hybrid --lr 0.00016 --root <data_dir>/car196```
### SOP
```python main.py --dataset SOP --loss hybrid --lr 4e-05 --batch_size 128 --root <data_dir>/Stanford_Online_Products```
### iNaturalist
```python main.py --dataset iNat --loss hybrid --lr 8e-05 --batch_size 256 --root <data_dir>/train_mini_resized```
### In-Shop
```python main.py --dataset inshop --loss hybrid --lr 4e-05 --batch_size 128 --root <data_dir>/inshop```

## Other Notes
`ablations.py` contains the code to run most of our ablation results. 

`toy_experiment.ipynb` contains the code for our simple toy experiment.

`sample_outputs` contains sample outputs of `main.py`.

Saved models of our implementation for all four datasets can be found here: [link](https://github.com/Chris210634/metric-learning-using-contextual-similarity/releases/tag/v1.0.0).

`data/*.txt` are the data-splits we used.

`data/create_datasplits_iNat.ipynb` contains the code we used to randomly split iNaturalist evenly into train and test sets. You do not need to run this, just use the same splits we used.
