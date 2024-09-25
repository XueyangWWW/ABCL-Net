import torch.nn

from models.base_model import BaseModel


def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'gen2seg_model_train':
        assert(opt.dataset_mode == 'gen2seg_train')
        from .gen2seg_model import GEN2SEGModel_TRAIN
        model = GEN2SEGModel_TRAIN(opt)

    elif opt.model == 'gen2seg_model_test':
        assert(opt.dataset_mode == 'gen2seg_test')
        from .gen2seg_model import GEN2SEGModel_TEST
        model = GEN2SEGModel_TEST(opt)

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.__init__(opt)
    return model


