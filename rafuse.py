import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *


class RafuseRecognize():
    
    def __init__(self):
        
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', 
                                model_dir = '/tmp/imagenet')
        
        
    def init_classify_image_model(self):
        
        create_graph('/tmp/imagenet')

        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        
        
    def recognize_image(self, image_data):
        
        predictions = self.sess.run(self.softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            #print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))
            #print(human_string)
            classification = self.refuse_classification.predict(human_string)
            result_list.append('%s  =>  %s' % (human_string, classification))
            
        return '\n'.join(result_list)
        

if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RafuseRecognize()
        image_data = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        res = test.recognize_image(image_data)
        print('classify:\n%s' %(res))
