# 快速搭建垃圾分类模型


## 下载模型

    python classify_image.py
    
    
    
## 测试模型

    python classify_image.py --image_file ./img/2.png 
    
    
    

## 垃圾分类标准

    ./data/refuse_classification.txt
    
    
    
## imagenet类别映射表

    
    - 英文对照表
    
        ./data/imagenet_synset_to_human_label_map.txt
    
    - 中文对照表
    
        ./data/imagenet_2012_challenge_label_chinese_map.pbtxt
    
    - id对照表
        
        ./data/imagenet_2012_challenge_label_map_proto.pbtxt
        
        
        

## 图像分类模型
    
    inception-v3
    


## 垃圾分类映射

    - 数据标注
        
        训练数据：./data/train_data.txt
        测试数据：./data/vilid_data.txt
        
    - 模型
    
        TextCNN
        
        详解: ./textcnn/README.md
        
        
        
## 垃圾分类识别

    - 识别
        python rafuse.py img/2.png
        
        输出结果：
            移动电话手机  =>  可回收垃圾
            iPod  =>  湿垃圾
            笔记本笔记本电脑  =>  可回收垃圾
            调制解调器  =>  湿垃圾
            手持电脑手持微电脑  =>  可回收垃圾

## 参考资料
https://blog.csdn.net/zengNLP/article/details/94783092
https://bbs.cvmart.net/topics/3414/created_at?
https://bbs.cvmart.net/topics/3414
https://bbs.cvmart.net/topics/3405
https://bbs.cvmart.net/topics/3416