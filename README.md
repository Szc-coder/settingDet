# settingDet
这个rep是完成信息系统工程课程的作业情况下，顺手提交rep供大家学习。
# 功能
实现了检测用户的当前姿态，目前只有两个比较粗糙的检测方案，都是采用计算向量夹角的方式进行检测。有兴趣的可以优化检测方案，或者自己创建数据集在模型后面添加分类头重新训练（该方案不确定有效）
# MODEL
使用了PaddlePaddle新推出的PP-TinyPose，单人场景FP16推理可达122FPS、51.8AP。测试时使用笔记本的摄像头基本够用。
# 环境
paddle 2.3\
paddleDet 2.4\
cuda 10.2\
python 3.7
# 文件
source -- paddleDet提供的python推理工具\
detModel -- PP-TinyPose的推理模型\
video_process --程序入口
