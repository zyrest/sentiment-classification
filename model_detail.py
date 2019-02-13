import os

import tensorflow as tf
from tensorflow import saved_model as sm

pd_dir = 'model/saved_model'
pd_path = os.path.join(pd_dir, 'best_validation')

session = tf.Session()
session.run(tf.global_variables_initializer())
meta_graph_def = sm.loader.load(session, tags=[sm.tag_constants.TRAINING], export_dir=pd_path)
graph_def = session.graph_def

for node in graph_def.node:
    print(node.name)
