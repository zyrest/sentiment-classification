import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TextCNNConfig, TextCNN
from data.weibo_loader import read_category, read_vocab
from tensorflow import saved_model as sm

base_dir = 'data/weibo'
vocab_dir = os.path.join(base_dir, 'weibo.vocab.txt')

save_dir = 'model/ckpt'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

pd_dir = 'model/saved_model'
pd_path = os.path.join(pd_dir, 'best_validation')

signature_key = sm.signature_constants.CLASSIFY_INPUTS


class CnnModel:
    def __init__(self):
        self.config = TextCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.meta_graph_def = sm.loader.load(self.session, tags=[sm.tag_constants.TRAINING], export_dir=pd_path)
        signature = self.meta_graph_def.signature_def

        x_tensor_name = signature[signature_key].inputs['input_x'].name
        kp_tensor_name = signature[signature_key].inputs['keep_prob'].name
        y_tensor_name = signature[signature_key].outputs['output'].name
        prob_tensor_name = signature[signature_key].outputs['prob'].name

        self.x = self.session.graph.get_tensor_by_name(x_tensor_name)
        self.kp = self.session.graph.get_tensor_by_name(kp_tensor_name)
        self.y = self.session.graph.get_tensor_by_name(y_tensor_name)
        self.prob = self.session.graph.get_tensor_by_name(prob_tensor_name)

    def predict(self, message):
        content = str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.kp: 1.0
        }

        y_pred_cls = self.session.run(self.prob, feed_dict=feed_dict)
        return y_pred_cls


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_strs = ['喜欢导演的叙事风格，通过音乐演员的动作所传达的是一种不能用语言描述的感觉',
                 '挂挡入位，倒车，行车，等操控都还不错',
                 'C罗能留在皇马所有人都开心。皇马球迷开心，C罗球迷开心，巴萨球迷也开心。',
                 '吃了几次，超喜欢，味道不错，特别是牛排；当然吃完还不用自己埋单，那简直就是人间美事了。',
                 '环境是不错，配套还需完善，园中人工的痕迹太过于严重，希望能领略更自然的景色']
    for i in test_strs:
        print(cnn_model.predict(i))
