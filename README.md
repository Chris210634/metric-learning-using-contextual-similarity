# Supervised Metric Learning for Retrieval via Contextual Similarity Optimization

![](https://github.com/Chris210634/metric-learning-using-contextual-similarity/raw/main/jaccard.png)

## Downloading Data
Run `download_data.sh` to download all four datasets. Note that there are two places you need to edit before running this script: (1) replace `<data_dir>` with the location where you want the data to live, generally speaking you want fast I/O to this location, e.g. scratch disk; (2) replace `<git_dir>` with the location of your local git clone (this directory). SOP, CUB and Cars are fast; iNaturalist will take 2-4 hours to download and process. For In-Shop, you must download the images manually from the authors' link (img.zip not img_highres.zip)

## Running our Algorithm
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

Saved models of our implementation for all four datasets can be found here: [link] (https://github.com/Chris210634/metric-learning-using-contextual-similarity/releases/tag/v1.0.0).

`data/*.txt` are the data-splits we used.

`data/create_datasplits_iNat.ipynb` contains the code we used to randomly split iNaturalist evenly into train and test sets. You do not need to run this, just use the same splits we used.
