from .base import *

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None):
        _txt_file = 'Ebay_train.txt' if mode == 'train' else 'Ebay_test.txt' 
        self.root = root
        self.mode = mode
        self.transform = transform
        self.super_labels = []

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata = open(os.path.join(self.root, _txt_file))
        for i, (image_id, class_id, super_class_id, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                # if int(class_id)-1 in self.classes:
                self.ys += [int(class_id)-1]
                self.super_labels += [int(super_class_id)-1]
                self.I += [int(image_id)-1]
                self.im_paths.append(os.path.join(self.root, path))