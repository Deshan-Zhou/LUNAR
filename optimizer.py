from settings import *
parm = get_settings()

class Optimizer(object):
    def __init__(self,reconstruct_loss,reconstruct_result,all_atten_parm):
        self.reconstruct_loss = reconstruct_loss
        self.reconstruct_result = reconstruct_result
        self.all_atten_parm = all_atten_parm
        self._build()

    def _build(self):

        optimize1 = tf.train.AdamOptimizer(learning_rate=parm.reconstruct_lr)
        gradients1,variables1 = zip(*optimize1.compute_gradients(self.reconstruct_loss))
        gradients1,_ = tf.clip_by_global_norm(gradients1,1)
        self.optimizer1 = optimize1.apply_gradients(zip(gradients1,variables1))

