import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_dg', 708, 'The number of Drug node')
flags.DEFINE_integer('num_pt', 1512, 'The number of Protein node')
flags.DEFINE_integer('num_ds', 5603, 'The number of Disease node')
flags.DEFINE_integer('num_se', 4192, 'The number of Side-effect node')

flags.DEFINE_integer('dim_dg', 1024, 'The dimension of Drug embedding')
flags.DEFINE_integer('dim_pt', 1024, 'The dimension of Protein embedding')
flags.DEFINE_integer('dim_ds', 1024, 'The dimension of Disease embedding')
flags.DEFINE_integer('dim_se', 1024, 'The dimension of Side-effect embedding')

flags.DEFINE_integer('dim_hid1', 1024, 'The dimension of Hidden1 Layer')
flags.DEFINE_integer('dim_hid2', 1024, 'The dimension of Hidden2 Layer')
flags.DEFINE_integer('dim_hid3', 1024, 'The dimension of Hidden2 Layer')
flags.DEFINE_integer('dim_proj1', 512, 'The dimension of Project matrix1')

flags.DEFINE_float('reconstruct_lr', 0.001, 'The learning rate for reconstruct loss')
flags.DEFINE_float('dropout', 0.0,'The prossibility of miss')

flags.DEFINE_integer('step', 1, 'The number of Train step')

flags.DEFINE_string('path', 'data/', 'The relative path of data')

def get_settings():
    return FLAGS