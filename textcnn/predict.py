import tensorflow as tf
import numpy as np
import os, sys
import data_input_helper as data_helpers
import jieba

# Parameters

# Data Parameters
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


class RefuseClassification():

    def __init__(self):
    
        self.w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)#加载词向量
        self.init_model()
        self.refuse_classification_map = {0: '可回收垃圾', 1: '有害垃圾', 2: '湿垃圾', 3: '干垃圾'}
        
        
    def deal_data(self, text, max_document_length = 10):
        
        words = jieba.cut(text)
        x_text = [' '.join(words)]
        x = data_helpers.get_text_idx(x_text, self.w2v_wr.model.vocab_hash, max_document_length)

        return x


    def init_model(self):
        
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                              allow_soft_placement=FLAGS.allow_soft_placement, 
                              log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            self.sess.as_default()
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

            # Get the placeholders from the graph by name
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
          
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                
    
    def predict(self, text):
    
        x_test = self.deal_data(text, 5)
        predictions = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
        
        refuse_text = self.refuse_classification_map[predictions[0]]
        return refuse_text


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RefuseClassification()
        res = test.predict(sys.argv[1])
        print('classify:', res)
