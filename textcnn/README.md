# TextCNN
基于tensorflow 实现


- 环境要求

    python: 3.x 
    tensorflow: 1.x
    jieba
    word2vec
 

- 模型训练

    python textcnn/train.py 
    
    训练数据：./data/train_data.txt
        
    
- 模型评估
    
    python textcnn/eval.py 
    
    测试数据：./data/vilid_data.txt


- 单句测试

    python textcnn/predict.py '电池'
    
    输出结果：
        classify: 有害垃圾
        
        
        
