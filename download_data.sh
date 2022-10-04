cd <data_dir> # you need to decide where to put the data
wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
wget http://ai.stanford.edu/~jkrause/car196/car_ims.tgz
tar -xvf car_ims.tgz
tar -xvf CUB_200_2011.tgz
unzip Stanford_Online_Products.zip
mkdir car196
mv car_ims car196

# replace <git_dir> with the absolute path to the local git repo
cp <git_dir>/data/cub200_train.txt CUB_200_2011
cp <git_dir>/data/cub200_test.txt CUB_200_2011
cp <git_dir>/data/cars196_train.txt car196
cp <git_dir>/data/cars196_test.txt car196

# This concludes the SOP, CUB, and Cars download steps.
# The following commands download, extract, and resize the iNat images
# These steps take an hour or two. maybe even a few hours
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
tar -xvf train_mini.tar.gz
cp <git_dir>/data/iNaturalist_train.txt .
cp <git_dir>/data/iNaturalist_test.txt .
cp <git_dir>/data/resize_iNat.py .
mkdir train_mini_resized
python resize_iNat.py # this script resizes all the images to 288 pixels and puts them in train_mini_resized
mv iNaturalist_train.txt train_mini_resized
mv iNaturalist_test.txt train_mini_resized

# in-shop: you need to download img.zip manually from the Google Drive directory on the dataset authors' website
# save img.zip to <data_dir>
unzip img.zip
mkdir inshop
mv img inshop
cp <git_dir>/data/inshop*.txt inshop