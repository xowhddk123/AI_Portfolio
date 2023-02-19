from cv_lib import gc, os, torch
from Utils import (
    model_path, main_path, set_seed, seed
)
from Models import Unet
from Evaluation import Evaluation
from Train import Trainer

set_seed(seed)

if __name__ =="__main__":

    def load(model):
        try:
            net = Unet()
            net.load_state_dict(torch.load(os.path.join(model_path, model+'.pt')))
            return net

        except:
            net = Unet()
            return net

    models = ['dent', 'scratch', 'spacing']

    for model in models:
        net = load(model)
        path = os.path.join(main_path, model)  # 모델 경로 지정
        Trainer = Trainer(net=net, path = path)  # 모델 학습
        Trainer.train(model)

        gc.collect()
        torch.cuda.empty_cache()

        del Trainer, net
    
    for model in models:
        path = os.path.join(main_path, model)
        test = Evaluation(os.path.join(model_path, model+'.pt'), path)
        test.validation()

        gc.collect()
        torch.cuda.empty_cache()

        del test
    