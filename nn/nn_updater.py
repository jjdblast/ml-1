import numpy as np


class SGDUpdater(object):
    def __init__(self, base_lr=0.1, decay=0.0, momentum=0.0, schedule_type=0, update_type=0):
        self.base_lr_ = base_lr
        self.decay_ = decay
        self.momentum_ = momentum
        self.schedule_type_ = schedule_type
        self.lr_ = 0.0
        self.w_momentum_ = []
        self.w_grad_ss_ = []
        self.update_type_ = update_type

    def register(self, w):
        self.w_momentum_.append(np.zeros(w.shape))
        self.w_grad_ss_.append(np.zeros(w.shape))
        return len(self.w_momentum_) - 1

    def schedule(self, n_iter_passed):
        if self.schedule_type_ == 0:
            self.lr_ = self.base_lr_
        else:
            pass

    def update(self, updater_id, w, wgrad):
        if self.update_type_ == 0:
            self.w_momentum_[updater_id] *= self.momentum_
            self.w_momentum_[updater_id] += self.lr_ * (wgrad + self.decay_*w)
            w -= self.w_momentum_[updater_id]
        elif self.update_type_ == 1:
            self.w_grad_ss_[updater_id] += wgrad**2
            adagrad = wgrad / (np.sqrt(self.w_grad_ss_[updater_id] + 1e-8))
            w -= self.lr_ * adagrad
        return w
