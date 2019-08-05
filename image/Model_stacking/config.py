#coding=utf-8
import warnings

class DefaultConfigs(object):
    env='default'
    model_name = "multimodal"
    train_data = "/home/liangc/URFC-2019/image/data/train/" # where is your train images data
    test_data = "/home/liangc/URFC-2019/image/data/test/"   # your test data
    load_model_path = None
    
    weights = "/home/liangc/URFC-2019/image/checkpoints/"
    best_models = "/home/liangc/URFC-2019/image/checkpoints/best_models/"
    debug_file='/home/liangc/URFC-2019/image/tmp/debug'
    submit = "/home/liangc/URFC-2019/image/submit/"
    
    num_classes = 9
    img_weight = 100
    img_height = 100
    channels = 3

    lr = 0.002
    lr_decay = 0.5
    weight_decay =0e-5
    batch_size = 64
    epochs = 16
    
def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
