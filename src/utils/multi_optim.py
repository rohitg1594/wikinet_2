from src.utils.utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MuliOptim(object):

    def __init__(self, args=None, model=None):

        self.emb_flag = False
        self.other_flag = False

        optimizer_type = get_optim(optim=args.embs_optim)
        embs_param = filter_embs_param(model)
        if not embs_param:
            self.emb_flag = True
            if args.sparse:
                self.emb_optimizer = optimizer_type(embs_param, lr=args.lr)
            else:
                self.emb_optimizer = optimizer_type(filter_embs_param(model), lr=args.lr, weight_decay=args.wd)
            self.emb_scheduler = ReduceLROnPlateau(self.emb_optimizer, mode='max', verbose=True, patience=5)

        optimizer_type = get_optim(optim=args.other_optim)
        other_param = filter_embs_param(model)
        if not other_param:
            self.other_flag = True
            self.other_optimizer = optimizer_type(filter_other_param(model), lr=args.lr, weight_decay=args.wd)
            self.other_scheduler = ReduceLROnPlateau(self.other_optimizer, mode='max', verbose=True, patience=5)

    def zero_grad(self):
        if self.emb_flag:
            self.emb_optimizer.zero_grad()
        if self.other_flag:
            self.other_optimizer.zero_grad()

    def optim_step(self):
        if self.emb_flag:
            self.emb_optimizer.step()
        if self.other_flag:
            self.other_optimizer.step()

    def scheduler_step(self, metric):
        if self.emb_flag:
            self.emb_scheduler.step(metric)
        if self.other_flag:
            self.other_scheduler.step(metric)

    def get_state_dict(self):
        state_dict = {}
        if self.emb_flag:
            state_dict['emb_optimizer'] = self.emb_optimizer.state_dict()
        if self.other_flag:
            state_dict['other_optimizer'] = self.other_optimizer.state_dict()

        return state_dict
