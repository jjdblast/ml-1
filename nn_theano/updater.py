from theano import function, tensor, shared
import numpy as np


class SGDUpdater(object):
    def __init__(self, base_lr=0.1, l2_decay=0.0, momentum=0.0,
                 tune_lr_type=0, lr_gamma=1.0, update_type=0):
        self.base_lr_ = base_lr
        self.l2_decay_ = l2_decay
        self.m_ = momentum
        self.tune_lr_type_ = tune_lr_type
        self.lr_gamma_ = lr_gamma
        self.update_type_ = update_type

        self.current_lr_ = self.base_lr_
        self.wms_ = []
        self.wg_sqr_sums_ = []

    def register_updater(self, w):
        shape = w.get_value().shape
        self.wms_.append(shared(np.zeros(shape)))
        self.wg_sqr_sums_.append(shared(np.zeros(shape)))
        return len(self.wms_) - 1

    def tune_lr(self, i_epoch):
        if self.tune_lr_type_ == 0:
            self.current_lr_ = self.base_lr_
        elif self.tune_lr_type_ == 1:
            self.current_lr_ = self.base_lr_ * tensor.pow(self.lr_gamma_, i_epoch)

    def update(self, w_wg_ids):
        updates = []
        for w, wg, id_ in w_wg_ids:
            if self.update_type_ == 0:
                w_update = w - self.current_lr_ * wg
                updates.append((w, w_update))
            elif self.update_type_ == 1:
                wm = self.wms_[id_] * self.m_
                wm_update = wm + self.current_lr_ * (wg + self.l2_decay_ * w)
                w_update = w - wm_update
                updates.extend([(w, w_update), (self.wms_[id_], wm_update)])
            elif self.update_type_ == 2:
                wg_sqr_sum_update = self.wg_sqr_sums_[id_] + wg**2
                adagrad = wg / (tensor.sqrt(wg_sqr_sum_update) + 1e-8)
                w_update = w - self.current_lr_ * adagrad
                updates.extend([(w, w_update), (self.wg_sqr_sums_[id_], wg_sqr_sum_update)])

        return updates
