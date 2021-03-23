## CVPR_2021_Papers汇总，主要包括论文链接、代码地址、文章解读等等
## 关注公众号【深度学习技术前沿】后台回复 **CVPR2021** 获得百度云下载链接

------

- 官网链接：http://cvpr2021.thecvf.com<br>
- 时间：2021年6月19日-6月25日<br>
- 论文接收公布时间：2021年2月28日<br>
- CVPR2021官方接受论文列表：[http://cvpr2021.thecvf.com/sites/default/files/2021-03/accepted_paper_ids.txt](http://cvpr2021.thecvf.com/sites/default/files/2021-03/accepted_paper_ids.txt)
------

# CVPR2021接受论文/代码分方向整理(持续更新)

# 分类目录：
## Low-Level-Vision(主要包括：超分辨率，图像恢复，去雨，去雾，去模糊，去噪，重建等方向)
- [1.超分辨率（Super-Resolution）](#1.超分辨率)
- [2.图像去雨（Image Deraining）](#2.图像去雨)
- [3.图像去雾（Image Dehazing）](#3.图像去雾)
- [4.去模糊（Deblurring）](#4.去模糊)
- [5.去噪（Denoising）](#5.去噪)
- [6.图像恢复（Image Restoration）](#6.图像恢复)
- [7.图像增强（Image Enhancement）](#7.图像增强)
- [8.图像去摩尔纹（Image Demoireing）](#8.图像去摩尔纹)
- [9.图像阴影去除(Image Shadow Removal)](#9.图像阴影去除)
- [10.图像翻译（Image Translation）](#10.图像翻译)
- [11.插帧（Frame Interpolation）](#11.插帧)
- [12.视频压缩（Video Compression）](#12.视频压缩)
- [13.图像编辑(Image Edit)](#13.图像编辑)

## High-Level-Vision（主要包括：图像分类，检测，分割，跟踪，GAN等方向）
### [检测](#detection)
* [图像目标检测(Image Object Detection)](#IOD)<br>
* [视频目标检测(Video Object Detection)](#VOD)<br>
* [三维目标检测(3D Object Detection)](#3DOD)<br>
* [动作检测(Activity Detection)](#ActivityDetection)<br>
* [异常检测(Anomally Detetion)](#AnomallyDetetion)<br>

### [图像分割(Image Segmentation)](#ImageSegmentation)
* [全景分割(Panoptic Segmentation)](#PanopticSegmentation)<br>
* [语义分割(Semantic Segmentation)](#SemanticSegmentation)<br>
* [实例分割(Instance Segmentation)](#InstanceSegmentation)<br>
* [超像素(Superpixel)](#Superpixel)<br>
* [视频目标分割(Video Object Segmentation)](#VOS)<br>
* [抠图(Matting)](#Matting)<br>

### [人脸(Face)](#Face)
* [人脸生成/合成/伪造(Face Generation/Face Synthesis/Face Forgery)](#FaceSynthesis)
* [人脸反欺骗(Face Anti-Spoofing)](#FaceAnti-Spoofing)

### [目标跟踪(Object Tracking)](#ObjectTracking)

### [重识别(Re-Identification)](#Re-Identification)
* [行为识别/行为识别/动作识别/检测/分割(Action/Activity Recognition)](#ActionRecognition)

### [医学影像(Medical Imaging)](#MedicalImaging)

### [GAN/生成式/对抗式(GAN/Generative/Adversarial)](#GAN)

### [估计(Estimation)](#Estimation)
* [人体姿态估计(Human Pose Estimation)](#HumanPoseEstimation)
* [手势估计(Gesture Estimation)](#GestureEstimation)
* [光流/位姿/运动估计(Flow/Pose/Motion Estimation)](#Flow/Pose/MotionEstimation)

### [三维视觉(3D Vision)](#3DVision)
* [三维点云(3D Point Cloud)](#3DPC)<br>
* [三维重建(3D Reconstruction)](#3DReconstruction)<br>

## 模型架构与数据处理（主要包括：Transformer, NAS，模型压缩，模型评估）
### [神经网络架构(Neural Network Structure)](#NNS)
* [图神经网络(GNN)](#GNN)<br>

### [Transformer](#att)

### [神经网络架构搜索(NAS)](#NAS)

### [数据处理(Data Processing)](#DataProcessing)
* [数据增广(Data Augmentation)](#DataAugmentation)<br>
* [归一化/正则化(Batch Normalization)](#BatchNormalization)<br>
* [图像聚类(Image Clustering)](#ImageClustering)<br>

### [模型压缩(Model Compression)](#ModelCompression)
* [知识蒸馏(Knowledge Distillation)](KnowledgeDistillation)<br>

### [模型评估(Model Evaluation)](#ModelEvaluation)

### [数据集(Database)](#Database)
<br>

## 其它方向
### [主动学习(Active Learning)](#ActiveLearning)

### [小样本学习/零样本(Few-shot Learning)](#Few-shotLearning)

### [持续学习(Continual Learning/Life-long Learning)](#ContinualLearning)

### [视觉推理(Visual Reasoning)](#VisualReasoning)

### [迁移学习/domain/自适应](#domain)

### [对比学习(Contrastive Learning)](#ContrastiveLearning)

### [图像/视频检索(Image Retrieval)](#ImageRetrieval)

#### [暂无分类](#100)
<br>

## CVPR2021的论文解读汇总
 - [论文解读](#300)

<br><br>

<a name="1.超分辨率"></a>

## 1.超分辨率（Super-Resolution）

### Unsupervised Degradation Representation Learning for Blind Super-Resolution
- Code：[https://github.com/LongguangWang/DASR](https://github.com/LongguangWang/DASR)

### Data-Free Knowledge Distillation For Image Super-Resolution

### Learning Continuous Image Representation with Local Implicit Image Function(通过局部隐含图像功能学习连续图像表示)
- [paepr](https://arxiv.org/abs/2012.09161) 
- [code](https://github.com/yinboc/liif) 
- [video](https://youtu.be/6f2roieSY_8) 
- [project](https://yinboc.github.io/liif/)

### AdderSR: Towards Energy Efficient Image Super-Resolution
- Paper：[https://arxiv.org/abs/2009.08891](https://arxiv.org/abs/2009.08891)
- [code](https://github.com/huawei-noah/AdderNet)
- 解读：[华为开源加法神经网络](https://zhuanlan.zhihu.com/p/113536045)

### Exploring Sparsity in Image Super-Resolution for Efficient Inference
- Paper：[https://arxiv.org/abs/2006.09603](https://arxiv.org/abs/2006.09603)
- Code：[https://github.com/LongguangWang/SMSR](https://github.com/LongguangWang/SMSR)

### ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic
- Code：[https://github.com/Xiangtaokong/ClassSR](https://github.com/Xiangtaokong/ClassSR)

### Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images
- Paper：[https://arxiv.org/abs/2011.14631](https://arxiv.org/abs/2011.14631)
- Homepage：[http://www.liuyebin.com/crossMPI/crossMPI.html](http://www.liuyebin.com/crossMPI/crossMPI.html)
- Analysis：[CVPR 2021，Cross-MPI以底层场景结构为线索的端到端网络，在大分辨率（x8）差距下也可完成高保真的超分辨率](https://zhuanlan.zhihu.com/p/354752197)

<a name="2.图像去雨"></a>
## 2.图像去雨（Image Deraining）
### Semi-Supervised Video Deraining with Dynamic Rain Generator(带动态雨水产生器的半监督视频去雨)
- [paper](https://arxiv.org/abs/2103.07939)

<a name="3.图像去雾"></a>
## 3.图像去雾（Image Dehazing）

<a name="4.去模糊"></a>
## 4.去模糊（Deblurring）
### DeFMO: Deblurring and Shape Recovery of Fast Moving Objects(快速移动物体的去模糊和形状恢复)
- [paper](https://arxiv.org/abs/2012.00595)
- [code](https://github.com/rozumden/DeFMO)
- [video](https://www.youtube.com/watch?v=pmAynZvaaQ4)

### ARVo: Learning All-Range Volumetric Correspondence for Video Deblurring(学习用于视频去模糊的全范围体积对应)
- [paper](https://arxiv.org/pdf/2103.04260.pdf)

<a name="5.去噪"></a>
## 5.去噪（Denoising）

<a name="6.图像恢复"></a>
## 6.图像恢复（Image Restoration）
### Multi-Stage Progressive Image Restoration
- Paper：[https://arxiv.org/abs/2102.02808](https://arxiv.org/abs/2102.02808)
- Code：[https://github.com/swz30/MPRNet](https://github.com/swz30/MPRNet)

### CT Film Recovery via Disentangling Geometric Deformation and Illumination Variation: Simulated Datasets and Deep Models
- Paper：[https://arxiv.org/abs/2012.09491](https://arxiv.org/abs/2012.09491)
- Code：[https://github.com/transcendentsky/Film-Recovery](https://github.com/transcendentsky/Film-Recovery)

### Generating Diverse Structure for Image Inpainting With Hierarchical VQ-VAE(使用分层VQ-VAE生成图像修复的多样结构)
- [paper](https://arxiv.org/pdf/2103.10022) 
- [code](https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting)

### PISE: Person Image Synthesis and Editing with Decoupled GAN(使用分离的GAN进行人像合成和编辑)
- [paper](https://arxiv.org/abs/2103.04023)
- [code](https://github.com/Zhangjinso/PISE)

### DeFLOCNet: Deep Image Editing via Flexible Low level Controls(通过灵活的低级控件进行深度图像编辑)

### PD-GAN: Probabilistic Diverse GAN for Image Inpainting(用于图像修复的概率多样GAN)

### Anycost GANs for Interactive Image Synthesis and Editing(用于交互式图像合成和编辑的AnyCost Gans)
- [paper](https://arxiv.org/abs/2103.03243) 
- [code](https://github.com/mit-han-lab/anycost-gan)

### Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing（利用GAN中潜在的空间维度进行实时图像编辑）

<a name="7.图像增强"></a>
## 7.图像增强（Image Enhancement）
### Auto-Exposure Fusion for Single-Image Shadow Removal
- Paper：[https://arxiv.org/abs/2103.01255](https://arxiv.org/abs/2103.01255)
- Code：[https://github.com/tsingqguo/exposure-fusion-shadow-removal](https://github.com/tsingqguo/exposure-fusion-shadow-removal)

### Learning Multi-Scale Photo Exposure Correction
- Paper：[https://arxiv.org/abs/2003.11596](https://arxiv.org/abs/2003.11596)
- Code：[https://github.com/mahmoudnafifi/Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)

### DeFMO: Deblurring and Shape Recovery of Fast Moving Objects
- Paper：[hhttps://arxiv.org/abs/2012.00595](https://arxiv.org/abs/2012.00595)
- Code：[https://github.com/rozumden/DeFMO](https://github.com/rozumden/DeFMO)

<a name="8.图像去摩尔纹"></a>
## 8.图像去摩尔纹（Image Demoireing）

<a name="9.图像阴影去除"></a>
## 9.图像阴影去除(Image Shadow Removal)
### Auto-Exposure Fusion for Single-Image Shadow Removal(用于单幅图像阴影去除的自动曝光融合)
 - [论文地址](https://arxiv.org/abs/2103.01255)
 - [代码地址](https://github.com/tsingqguo/exposure-fusion-shadow-removal)

<a name="10.图像翻译"></a>
## 10.图像翻译（Image Translation）
### Image-to-image Translation via Hierarchical Style Disentanglement
 - 论文地址：[https://arxiv.org/abs/2103.01456](https://arxiv.org/abs/2103.01456)
 - [代码地址](https://github.com/imlixinyang/HiSD)

### Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation(样式编码：用于图像到图像翻译的StyleGAN编码器)
- [paper](https://arxiv.org/abs/2008.00951)
- [code](https://github.com/eladrich/pixel2style2pixel)
- [project](https://eladrich.github.io/pixel2style2pixel/)

### CoMoGAN: continuous model-guided image-to-image translation(连续的模型指导的图像到图像翻译)
- [paper](https://arxiv.org/abs/2103.06879) 
- [code](http://github.com/cv-rits/CoMoGAN)

### Spatially-Adaptive Pixelwise Networks for Fast Image Translation(空间自适应像素网络，用于快速图像翻译)
- [paper](https://arxiv.org/abs/2012.02992) 
- [project](https://tamarott.github.io/ASAPNet_web/)

<a name="11.插帧"></a>
## 11.插帧（Frame Interpolation）
### FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
- Paper：[https://arxiv.org/abs/2012.08512](https://arxiv.org/abs/2012.08512)
- Code：[https://tarun005.github.io/FLAVR/Code](https://tarun005.github.io/FLAVR/Code)
- Homepage：[https://tarun005.github.io/FLAVR/](https://tarun005.github.io/FLAVR/)

### CDFI: Compression-driven Network Design for Frame Interpolation
- Code:[https://github.com/tding1/Compression-Driven-Frame-Interpolation](https://github.com/tding1/Compression-Driven-Frame-Interpolation)

### DeFMO: Deblurring and Shape Recovery of Fast Moving Objects
- Paper：[hhttps://arxiv.org/abs/2012.00595](https://arxiv.org/abs/2012.00595)
- Code：[https://github.com/rozumden/DeFMO](https://github.com/rozumden/DeFMO)

<a name="12.视频压缩"></a>
## 12.视频压缩（Video Compression）
### MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing
- Paper：[https://arxiv.org/abs/2103.01786](https://arxiv.org/abs/2103.01786)
- Code：[https://github.com/xyvirtualgroup/MetaSCI-CVPR2021](https://github.com/xyvirtualgroup/MetaSCI-CVPR2021)

<a name="13.图像编辑"></a>
## 13.图像编辑(Image Edit)
### Anycost GANs for Interactive Image Synthesis and Editing(用于交互式图像合成和编辑的AnyCost Gans)
- [paper](https://arxiv.org/abs/2103.03243)
- [code](https://github.com/mit-han-lab/anycost-gan)
### Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing（利用GAN中潜在的空间维度进行实时图像编辑）

<br>
<a name="detection"/> 

# 检测

<a name="IOD"/> 

### 图像目标检测(Image Object Detection)

[1] [Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection(小样本目标检测的语义关系推理)](https://arxiv.org/abs/2103.01903)

[2] [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/pdf/2011.09094.pdf)
  - 解读：[无监督预训练检测器](https://www.zhihu.com/question/432321109/answer/1606004872)
  
[3] Positive-Unlabeled Data Purification in the Wild for Object Detection(野外检测对象的阳性无标签数据提纯)

[4] [General Instance Distillation for Object Detection(通用实例蒸馏技术在目标检测中的应用)](https://arxiv.org/abs/2103.02340)

[5] [Instance Localization for Self-supervised Detection Pretraining(自监督检测预训练的实例定位)](https://arxiv.org/pdf/2102.08318.pdf)
  - [code](https://github.com/limbo0000/InstanceLoc)

[6] Multiple Instance Active Learning for Object Detection（用于对象检测的多实例主动学习)
- 论文地址: [https://github.com/yuantn/MIAL/raw/master/paper.pdf](https://github.com/yuantn/MIAL/raw/master/paper.pdf)
- 代码地址：[https://github.com/yuantn/MIAL](https://github.com/yuantn/MIAL)

[7] Towards Open World Object Detection(开放世界中的目标检测)
- [code](https://github.com/JosephKJ/OWOD)

[8] You Only Look One-level Feature
- [paper](https://arxiv.org/pdf/2103.09460.pdf)
- [code](https://github.com/megvii-model/YOLOF)

[9] End-to-End Object Detection with Fully Convolutional Network()
- [paper](https://arxiv.org/abs/2012.03544) 
- [code](https://github.com/Megvii-BaseDetection/DeFCN)
- 解读：[丢弃Transformer，FCN也可以实现E2E检测](https://zhuanlan.zhihu.com/p/332281368)

[10] FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding(通过对比提案编码进行的小样本目标检测)
- [paper](https://arxiv.org/abs/2103.05977)

[11] Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection(学习可靠的定位质量估计用于密集目标检测)
- [paper](https://arxiv.org/pdf/2011.12885.pdf) 
- [code](https://github.com/implus/GFocalV2) 
- 解读:[大白话 Generalized Focal Loss V2](https://zhuanlan.zhihu.com/p/313684358)

[12] MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection(用于类别识别无监督域自适应对象检测)
- [paper](https://arxiv.org/pdf/2103.04224.pdf)

[13] OPANAS: One-Shot Path Aggregation Network Architecture Search for Object(一键式路径聚合网络体系结构搜索对象)
- [paper](https://arxiv.org/abs/2103.04507) 
- [code](https://github.com/VDIGPKU/OPANAS)

[14] UP-DETR: Unsupervised Pre-training for Object Detection with Transformers
- [paper](https://arxiv.org/pdf/2011.09094.pdf) 
- [code](https://github.com/dddzg/up-detr)
- 解读：[无监督预训练检测器](https://www.zhihu.com/question/432321109/answer/1606004872)

<a name="VOD"/> 

### 视频目标检测(Video Object Detection)

[1] Depth from Camera Motion and Object Detection(相机运动和物体检测的深度)
- [论文地址](https://arxiv.org/abs/2103.01468)

[2] There is More than Meets the Eye: Self-Supervised Multi-Object Detection  and Tracking with Sound by Distilling Multimodal Knowledge(多模态知识提取的自监督多目标检测与有声跟踪)
- 论文地址:[https://arxiv.org/abs/2103.01353](https://arxiv.org/abs/2103.01353)
- [视频链接](https://www.youtube.com/channel/UCRpM8k1GY3kD2TqCo_yKN3g)
- [project](http://rl.uni-freiburg.de/research/multimodal-distill)

[3] Dogfight: Detecting Drones from Drone Videos（从无人机视频中检测无人机）

<a name="3DOD"/> 

### 三维目标检测(3D object detection)

[1] 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection(利用IoU预测进行半监督3D对象检测)
- [论文地址](https://arxiv.org/pdf/2012.04355.pdf)
- [代码地址](https://github.com/THU17cyz/3DIoUMatch)
- [项目地址](https://thu17cyz.github.io/3DIoUMatch/)
- [视频链接](https://youtu.be/nuARjhkQN2U)

[2] Categorical Depth Distribution Network for Monocular 3D Object Detection(用于单目三维目标检测的分类深度分布网络)
- [paper](https://arxiv.org/abs/2103.01100)

[3] ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection(ST3D：在三维目标检测上进行无监督域自适应的自训练)
- [paper](https://arxiv.org/pdf/2103.05346.pdf) 
- [code](https://github.com/CVMI-Lab/ST3D)

[4] Center-based 3D Object Detection and Tracking(基于中心的3D目标检测和跟踪)
- [paper](https://arxiv.org/abs/2006.11275) 
- [code](https://github.com/tianweiy/CenterPoint)

<a name="Activity Detection"/> 

### 动作检测(Activity Detection)

[1] Coarse-Fine Networks for Temporal Activity Detection in Videos
- [paper](https://arxiv.org/abs/2103.01302)

[2] Detecting Human-Object Interaction via Fabricated Compositional Learning(通过人为构图学习检测人与物体的相互作用)
- [paper](https://arxiv.org/abs/2103.08214) 
- [code](https://github.com/zhihou7/FCL)

[3] Reformulating HOI Detection as Adaptive Set Prediction(将人物交互检测重新配置为自适应集预测)
- [paper](https://arxiv.org/abs/2103.05983) 
- [code](https://arxiv.org/abs/2103.05983)

[4] QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information(具有图像范围的上下文信息的基于查询的成对人物交互检测)
- [paper](https://arxiv.org/abs/2103.05399) 
- [code](https://github.com/hitachi-rd-cv/qpic)

[5] End-to-End Human Object Interaction Detection with HOI Transformer(使用HOI Transformer进行端到端的人类对象交互检测)
- [paper](https://arxiv.org/pdf/2103.04503.pdf) 
- [code](https://github.com/bbepoch/HoiTransformer)

<a name="AnomallyDetetionn"/> 

### 异常检测(Anomally Detetion)

[1] Multiresolution Knowledge Distillation for Anomaly Detection(用于异常检测的多分辨率知识蒸馏)
- [paper](https://arxiv.org/abs/2011.11108)

[2] ReDet: A Rotation-equivariant Detector for Aerial Object Detection(ReDet：用于航空物体检测的等速旋转检测器)
- [paper](https://arxiv.org/abs/2103.07733) 
- [code](https://github.com/csuhan/ReDet)

[3] Dense Label Encoding for Boundary Discontinuity Free Rotation Detection(密集标签编码，用于边界不连续自由旋转检测)
- [paper](https://arxiv.org/abs/2011.09670) 
- [code](https://github.com/yangxue0827/RotationDetection) 
- [解读-DCL：旋转目标检测新方法](https://zhuanlan.zhihu.com/p/354373013)

[4] Skeleton Merger: an Unsupervised Aligned Keypoint Detector(骨架合并：无监督的对准关键点检测器)
- [paper](https://arxiv.org/pdf/2103.10814.pdf) 
- [code](https://github.com/eliphatfs/SkeletonMerger)

<br>

<a name="ImageSegmentation"/> 

## 图像分割(Image Segmentation)

[1] Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?
 - [paper](https://arxiv.org/abs/2012.06166)
 - [code](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation)

[2] PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation(语义流经点以进行航空图像分割)

[3] PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation(语义流经点以进行航空图像分割)
- [paper](https://arxiv.org/pdf/2103.06564.pdf)

[4] FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space(在连续频率空间中通过情景学习进行医学图像分割的联合域泛化)
- [paper](https://arxiv.org/abs/2103.06030) 
- [code](https://github.com/liuquande/FedDG-ELCFS)

<a name="PanopticSegmentation"/> 

### 全景分割(Panoptic Segmentation)

[1] Cross-View Regularization for Domain Adaptive Panoptic Segmentation(用于域自适应全景分割的跨视图正则化)
- [paper](https://arxiv.org/abs/2103.02584)

[2] 4D Panoptic LiDAR Segmentation（4D全景LiDAR分割）
- [paper](https://arxiv.org/abs/2102.12472)

<a name="SemanticSegmentation"/> 

### 语义分割(Semantic Segmentation)

[1] Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges(走向城市规模3D点云的语义分割：数据集，基准和挑战)
- [paper](https://arxiv.org/abs/2009.03137)
- [code](https://github.com/QingyongHu/SensatUrban)

[2] PLOP: Learning without Forgetting for Continual Semantic Segmentation（PLOP：学习而不会忘记连续的语义分割）
- [paper](https://arxiv.org/abs/2011.11390)

[3] Cross-Dataset Collaborative Learning for Semantic Segmentation(跨数据集协同学习的语义分割)
- [paper](https://arxiv.org/abs/2103.11351)

[4] BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation(用于弱监督语义和实例细分的边界框归因图)
- [paper](https://arxiv.org/abs/2103.08907)

[5] Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations(通过稀疏和纠缠的潜在表示的排斥力进行连续语义分割)
- [paper](https://arxiv.org/abs/2103.06342)

[6] Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion(通过双边扩充和自适应融合对实点云场景进行语义分割)
- [paper](https://arxiv.org/abs/2103.07074)

[7] Capturing Omni-Range Context for Omnidirectional Segmentation(捕获全方位上下文进行全方位分割)
- [paper](https://arxiv.org/abs/2103.05687)

[8] MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation(MetaCorrection：语义分割中无监督域自适应的域感知元丢失校正)
- [paper](https://arxiv.org/abs/2103.05254)

[9] Learning Statistical Texture for Semantic Segmentation(学习用于语义分割的统计纹理)
- [paper](https://arxiv.org/abs/2103.04133)

[10] Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation(基于双层域混合的半监督域自适应语义分割)
- [paper](https://arxiv.org/pdf/2103.04705.pdf)

[11] Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation(多源领域自适应与协作学习的语义分割)
- [paper](https://arxiv.org/abs/2103.04717)

<a name="InstanceSegmentation"/> 
### 实例分割(Instance Segmentation)

[1] End-to-End Video Instance Segmentation with Transformers(使用Transformer的端到端视频实例分割) 
- [paper](https://arxiv.org/abs/2011.14503)
- [code](https://github.com/Epiphqny/VisTR)

[2] BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation(用于弱监督语义和实例细分的边界框归因图)
- [paper](https://arxiv.org/abs/2103.08907)

<a name="Superpixel"/> 
## 超像素(Superpixel)

[1] Learning the Superpixel in a Non-iterative and Lifelong Manner(以非迭代和终身的方式学习超像素)
- [paper](https://arxiv.org/pdf/2103.10681.pdf)

<a name="VOS"/> 

### 视频目标分割(Video Object Segmentation)

[1] Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild(学习推荐帧用于交互式野外视频对象分割)
- [paper](https://arxiv.org/pdf/2103.10391.pdf) 
- [code](https://github.com/svip-lab/IVOS-W)

[2] Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion(模块化交互式视频对象分割：面具交互，传播和差异感知融合)
- [paper](https://arxiv.org/pdf/2103.07941.pdf) 
- [project](https://hkchengrex.github.io/MiVOS/)

<a name="Matting"/> 

## 抠图(Matting)

[1] Real-Time High Resolution Background Matting
- [paper](https://arxiv.org/abs/2012.07810)
- [code](https://github.com/PeterL1n/BackgroundMattingV2)
- [project](https://grail.cs.washington.edu/projects/background-matting-v2/)
- [video](https://youtu.be/oMfPTeYDF9g)

<a name="Estimation"/> 

## 9. 估计(Estimation)

<a name="HumanPoseEstimation"/> 

### 人体姿态估计(Human Pose Estimation)
[1] CanonPose: Self-supervised Monocular 3D Human Pose Estimation in the Wild（野外自监督的单眼3D人类姿态估计）

[2] PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers（具有透视作物层的3D姿势的几何感知神经重建）
- [paper](https://arxiv.org/abs/2011.13607)

[3] DCPose: Deep Dual Consecutive Network for Human Pose Estimation(用于人体姿态估计的深度双重连续网络)
- [paper](https://arxiv.org/abs/2103.07254) 
- [code](https://github.com/Pose-Group/DCPose)

[4] Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing(用于实例感知人类语义解析的可微分多粒度人类表示学习)
- [paper](https://arxiv.org/pdf/2103.04570.pdf) 
- [code](https://github.com/tfzhou/MG-HumanParsing)

<a name="Flow/Pose/MotionEstimation"/> 

### 手势估计(Gesture Estimation)

[1] Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive  2D-1D Registration(基于语义聚合和自适应2D-1D配准的相机空间手部网格恢复)
- [paper](https://arxiv.org/pdf/2103.02845.pdf)
- [code](https://github.com/SeanChenxy/HandMesh)

[2] Skeleton Based Sign Language Recognition Using Whole-body Keypoints(基于全身关键点的基于骨架的手语识别)
- [paper](https://arxiv.org/abs/2103.08833) 
- [code](https://github.com/jackyjsy/CVPR21Chal-SLR)

<a name="Flow/Pose/MotionEstimation"/> 

### 光流/位姿/运动估计(Flow/Pose/Motion Estimation)

[1] GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation(用于单眼6D对象姿态估计的几何引导直接回归网络)
- [paper](http://arxiv.org/abs/2102.12145)
- [code](https://github.com/THU-DA-6D-Pose-Group/GDR-Net)

[2] Robust Neural Routing Through Space Partitions for Camera Relocalization in Dynamic Indoor Environments(在动态室内环境中，通过空间划分的鲁棒神经路由可实现摄像机的重新定位)
- [paper](https://arxiv.org/abs/2012.04746)
- [project](https://ai.stanford.edu/~hewang/)

[3] MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization(通过3D扫描同步进行多主体分割和运动估计)
- [paper](https://arxiv.org/pdf/2101.06605.pdf)
- [code](https://github.com/huangjh-pub/multibody-sync)

<a name="Face"/> 

## 人脸(Face)

[1] Cross Modal Focal Loss for RGBD Face Anti-Spoofing(Cross Modal Focal Loss for RGBD Face Anti-Spoofing)
- [paper](https://arxiv.org/abs/2103.00948)

[2] When Age-Invariant Face Recognition Meets Face Age Synthesis: A  Multi-Task Learning Framework(当年龄不变的人脸识别遇到人脸年龄合成时：一个多任务学习框架)
- [paper](https://arxiv.org/abs/2103.01520)
- [code](https://github.com/Hzzone/MTLFace)

[3] Multi-attentional Deepfake Detection(多注意的深伪检测)
- [paper](https://arxiv.org/abs/2103.02406)

[4] Image-to-image Translation via Hierarchical Style Disentanglement
- [paper](https://arxiv.org/abs/2103.01456)
- [code](https://github.com/imlixinyang/HiSD)

[5] A 3D GAN for Improved Large-pose Facial Recognition(用于改善大姿势面部识别的3D GAN)
- [paper](https://arxiv.org/pdf/2012.10545.pdf)

<br>

<a name="ObjectTracking"/> 

## 目标跟踪(Object Tracking)

[1] HPS: localizing and tracking people in large 3D scenes from wearable sensors(通过可穿戴式传感器对大型3D场景中的人进行定位和跟踪)

[2] Track to Detect and Segment: An Online Multi-Object Tracker(跟踪检测和分段：在线多对象跟踪器)
- [project](https://jialianwu.com/projects/TraDeS.html)
- [video](https://www.youtube.com/watch?v=oGNtSFHRZJA)

[3] Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking(多目标跟踪的概率小波计分和修复)
- [paper](https://arxiv.org/abs/2012.02337)

[4] Rotation Equivariant Siamese Networks for Tracking（旋转等距连体网络进行跟踪）
- [paper](https://arxiv.org/abs/2012.13078)

[5] Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking(Transformer与追踪器相遇：利用时间上下文进行可靠的视觉追踪)
- [paper](https://arxiv.org/pdf/2103.11681)

[6] Track to Detect and Segment: An Online Multi-Object Tracker(跟踪检测和分段：在线多目标跟踪器)
- [paper](https://arxiv.org/abs/2103.08808) | [code](https://jialianwu.com/projects/TraDeS.html)

[7] Learning a Proposal Classifier for Multiple Object Tracking(用于多对象跟踪的分类器)
- [paper](https://arxiv.org/abs/2103.07889) 
- [code](https://github.com/daip13/LPC_MOT.git)

[8] Center-based 3D Object Detection and Tracking(基于中心的3D目标检测和跟踪)
- [paper](https://arxiv.org/abs/2006.11275) 
- [code](https://github.com/tianweiy/CenterPoint)

<br>

<a name="FaceSynthesis"/> 

### 人脸生成/合成/伪造(Face Generation/Face Synthesis/Face Forgery)

[1] Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection(【人脸伪造检测】由单中心损失监督的频率感知判别特征学习，用于人脸伪造检测)
- [paper](https://arxiv.org/abs/2103.09096)

[2] 3DCaricShop: A Dataset and A Baseline Method for Single-view 3D Caricature Face Reconstruction(单视图3D漫画面部重建的数据集和基线方法)
- [paper](https://arxiv.org/pdf/2103.08204.pdf) 
- [project](https://qiuyuda.github.io/3DCaricShop/)

[3] ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis(进行全面伪造分析的多功能基准)
- [paper](https://arxiv.org/abs/2103.05630) 
- [code](https://yinanhe.github.io/projects/forgerynet.html)

[4] Image-to-image Translation via Hierarchical Style Disentanglement(通过分层样式分解实现图像到图像的翻译)
- [paper](https://arxiv.org/abs/2103.01456) 
- [code](https://github.com/imlixinyang/HiSD)

[5] When Age-Invariant Face Recognition Meets Face Age Synthesis: A  Multi-Task Learning Framework(当年龄不变的人脸识别遇到人脸年龄合成时：一个多任务学习框架)<br>
- [paper](https://arxiv.org/abs/2103.01520) 
- [code](https://github.com/Hzzone/MTLFace)

[6] PISE: Person Image Synthesis and Editing with Decoupled GAN(使用分离的GAN进行人像合成和编辑)
- [paper](https://arxiv.org/abs/2103.04023) 
- [code](https://github.com/Zhangjinso/PISE)

[7] Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders(分析和改进自省变分自动编码器)
- [paper](https://arxiv.org/pdf/2012.13253.pdf) 
- [code](https://github.com/taldatech/soft-intro-vae-pytorch) 
- [project](https://taldatech.github.io/soft-intro-vae-web/)

<a name="FaceAnti-Spoofing"/> 

### 人脸反欺骗(Face Anti-Spoofing)

[1] Cross Modal Focal Loss for RGBD Face Anti-Spoofing(跨模态焦点损失，用于RGBD人脸反欺骗)
- [paper](https://arxiv.org/abs/2103.00948)

[2] Multi-attentional Deepfake Detection(多注意的Deepfake检测)
- [paper](https://arxiv.org/abs/2103.02406)

<br>

<a name="Re-Identification"/> 
## 重识别

[1] Meta Batch-Instance Normalization for Generalizable Person Re-Identification(通用批处理人员重新标识的元批实例规范化)
- [paper](https://arxiv.org/abs/2011.14670)


<a name="ImageRetrieval"/> 

## 图像/视频检索(Image/Video Retrieval)

[2] On Semantic Similarity in Video Retrieval(视频检索中的语义相似度)<br>
[paper](https://arxiv.org/abs/2103.10095) ｜ [code](https://mwray.github.io/SSVR/)<br><br>

[1] QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval(实用的查询高效的图像检索黑盒攻击)<br>
[paper](https://arxiv.org/abs/2103.02927)<br><br>

<a name="ActionRecognition"/> 

### 行为识别/动作识别/检测/分割(Action/Activity Recognition)

[1] Temporally-Weighted Hierarchical Clustering for Unsupervised Action Segmentation(临时加权层次聚类，实现无监督动作分割)
- [paper](https://arxiv.org/abs/2103.11264) 
- [code](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH)

[2] Coarse-Fine Networks for Temporal Activity Detection in Videos(粗细网络，用于视频中的时间活动检测)
- [paper](https://arxiv.org/abs/2103.01302)

[3] Learning Discriminative Prototypes with Dynamic Time Warping(通过动态时间扭曲学习判别性原型)
- [paper](https://arxiv.org/pdf/2103.09458.pdf)

[4] Temporal Action Segmentation from Timestamp Supervision(时间监督中的时间动作分割)
- [paper](https://arxiv.org/abs/2103.06669)

[5] ACTION-Net: Multipath Excitation for Action Recognition(用于动作识别的多路径激励)
- [paper](https://arxiv.org/abs/2103.07372) 
- [code](https://github.com/V-Sense/ACTION-Net)

[6] BASAR:Black-box Attack on Skeletal Action Recognition(骨骼动作识别的黑匣子攻击)
- [paper](https://arxiv.org/abs/2103.05266)

[7] Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack(了解对抗攻击下基于骨骼的动作识别的鲁棒性)
- [paper](https://arxiv.org/pdf/2103.05347.pdf)

[8] Temporal Difference Networks for Efficient Action Recognition(用于有效动作识别的时差网络)
- [paper](https://arxiv.org/abs/2012.10071) 
- [code](https://github.com/MCG-NJU/TDN)

[9] Behavior-Driven Synthesis of Human Dynamics(行为驱动的人类动力学综合)
- [paper](https://arxiv.org/pdf/2103.04677.pdf) 
- [code](https://compvis.github.io/behavior-driven-video-synthesis/)

<a name="MedicalImaging"/> 

## 医学影像(Medical Imaging)

[1] DeepTag: An Unsupervised Deep Learning Method for Motion Tracking on  Cardiac Tagging Magnetic Resonance Images(一种心脏标记磁共振图像运动跟踪的无监督深度学习方法)
- [paper](https://arxiv.org/abs/2103.02772)

[2] Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning(多机构协作改进基于深度学习的联合学习磁共振图像重建)
- [paper](https://arxiv.org/abs/2103.02148)
- [code](https://github.com/guopengf/FLMRCM)

[3] 3D Graph Anatomy Geometry-Integrated Network for Pancreatic Mass Segmentation, Diagnosis, and Quantitative Patient Management(用于胰腺肿块分割，诊断和定量患者管理的3D图形解剖学几何集成网络)

[4] Deep Lesion Tracker: Monitoring Lesions in 4D Longitudinal Imaging Studies(深部病变追踪器：在4D纵向成像研究中监控病变)
- [paper](https://arxiv.org/abs/2012.04872)

[5] Automatic Vertebra Localization and Identification in CT by Spine Rectification and Anatomically-constrained Optimization(通过脊柱矫正和解剖学约束优化在CT中自动进行椎骨定位和识别)
- [paper](https://arxiv.org/abs/2012.07947)

[6] Brain Image Synthesis with Unsupervised Multivariate Canonical CSCℓ4Net(无监督多元规范CSCℓ4Net的脑图像合成)
- [paper](https://arxiv.org/pdf/2103.11587.pdf)

[7] XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations(使用全局和局部解释诊断胸部X光片)
- [paper](https://arxiv.org/pdf/2103.10663.pdf)

[8] FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space(在连续频率空间中通过情景学习进行医学图像分割的联合域泛化)
- [paper](https://arxiv.org/abs/2103.06030) 
- [code](https://github.com/liuquande/FedDG-ELCFS)

[9] Multiple Instance Captioning: Learning Representations from Histopathology Textbooks and Articles(多实例字幕：从组织病理学教科书和文章中学习表示形式)
- [paper](https://arxiv.org/pdf/2103.05121.pdf)

[10] Discovering Hidden Physics Behind Transport Dynamics(在运输动力学背后发现隐藏物理)
- [paper](https://arxiv.org/abs/2011.12222)

<br>

<a name="NAS"/> 

## 神经网络架构搜索(NAS)

[1] AttentiveNAS: Improving Neural Architecture Search via Attentive(通过注意力改善神经架构搜索) 
- [paper](https://arxiv.org/pdf/2011.09011.pdf)

[2] ReNAS: Relativistic Evaluation of Neural Architecture Search(NAS predictor当中ranking loss的重要性)
- [paper](https://arxiv.org/pdf/1910.01523.pdf)

[3] HourNAS: Extremely Fast Neural Architecture Search Through an Hourglass Lens（降低NAS的成本）
- [paper](https://arxiv.org/pdf/2005.14446.pdf)

[4] Prioritized Architecture Sampling with Monto-Carlo Tree Search(蒙特卡洛树搜索的优先架构采样)
- [paper](https://arxiv.org/pdf/2103.11922.pdf) 
- [code](https://github.com/xiusu/NAS-Bench-Macro)

[5] Searching by Generating: Flexible and Efficient One-Shot NAS with Architecture Generator(通过生成进行搜索：带有架构生成器的灵活高效的一键式NAS)
- [paper](https://arxiv.org/abs/2103.07289) 
- [code](https://github.com/eric8607242/SGNAS)

[6] Contrastive Neural Architecture Search with Neural Architecture Comparators(带有神经结构比较器的对比神经网络架构搜索)
- [paper](https://arxiv.org/abs/2103.05471) 
- [code](https://github.com/chenyaofo/CTNAS)

[7] OPANAS: One-Shot Path Aggregation Network Architecture Search for Object(一键式路径聚合网络体系结构搜索对象)
- [paper](https://arxiv.org/abs/2103.04507) 
- [code](https://github.com/VDIGPKU/OPANAS)

<br>

<a name="GAN"/> 

## GAN/生成式/对抗式(GAN/Generative/Adversarial)

[1] Anycost GANs for Interactive Image Synthesis and Editing(用于交互式图像合成和编辑的AnyCost Gans)
- [paper](https://arxiv.org/abs/2103.03243)
- [code](https://github.com/mit-han-lab/anycost-gan)

[2] Efficient Conditional GAN Transfer with Knowledge Propagation across Classes(高效的有条件GAN转移以及跨课程的知识传播)
- [paper](高效的有条件GAN转移以及跨课程的知识传播)
- [code](http://github.com/mshahbazi72/cGANTransfer)

[3] Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing（利用GAN中潜在的空间维度进行实时图像编辑）

[4] Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs(Hijack-GAN：意外使用经过预训练的黑匣子GAN)
- [paper](https://arxiv.org/pdf/2011.14107.pdf)

[5] Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation(样式编码：用于图像到图像翻译的StyleGAN编码器)
- [paper](https://arxiv.org/abs/2008.00951)
- [code](https://github.com/eladrich/pixel2style2pixel)
- [project](https://eladrich.github.io/pixel2style2pixel/)

[6] A 3D GAN for Improved Large-pose Facial Recognition(用于改善大姿势面部识别的3D GAN)
- [paper](https://arxiv.org/pdf/2012.10545.pdf)

[7] DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network(通过对比生成对抗网络进行多种条件图像合成)
- [paper](https://arxiv.org/abs/2103.07893)

[8] Diverse Semantic Image Synthesis via Probability Distribution Modeling(基于概率分布建模的多种语义图像合成)
- [paper](https://arxiv.org/abs/2103.06878) 
- [code](https://github.com/tzt101/INADE.git)

[9] HumanGAN: A Generative Model of Humans Images(人类图像的生成模型)
- [paper](https://arxiv.org/abs/2103.06902)

[10] MetaSimulator: Simulating Unknown Target Models for Query-Efficient Black-box Attacks(模拟未知目标模型以提高查询效率的黑盒攻击)
- [paper](https://arxiv.org/abs/2009.00960) 
- [code](https://github.com/machanic/MetaSimulator)

[11] Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders(分析和改进自省变分自动编码器)
- [paper](https://arxiv.org/pdf/2012.13253.pdf) 
- [code](https://github.com/taldatech/soft-intro-vae-pytorch) \
- [project](https://taldatech.github.io/soft-intro-vae-web/)

[12] LOHO: Latent Optimization of Hairstyles via Orthogonalization(LOHO：通过正交化潜在地优化发型)
- [paper](https://arxiv.org/pdf/2103.03891.pdf)

[13] PISE: Person Image Synthesis and Editing with Decoupled GAN(使用分离的GAN进行人像合成和编辑)
- [paper](https://arxiv.org/abs/2103.04023) \
- [code](https://github.com/Zhangjinso/PISE)

[14] Closed-Form Factorization of Latent Semantics in GANs(GAN中潜在语义的闭式分解)
- [paper](https://arxiv.org/abs/2007.06600) 
- [code](https://github.com/genforce/sefa)

[15] PD-GAN: Probabilistic Diverse GAN for Image Inpainting(用于图像修复的概率多样GAN)


<a name="3DVision"/> 

## 三维视觉(3D Vision)

[2] A Deep Emulator for Secondary Motion of 3D Characters(三维角色二次运动的深度仿真器)
- [paper](https://arxiv.org/abs/2103.01261)

[1] 3D CNNs with Adaptive Temporal Feature Resolutions(具有自适应时间特征分辨率的3D CNN)
- [paper](https://arxiv.org/abs/2011.08652)

<a name="3DPC"/> 

### 三维点云(3D Point Cloud)

[14] Skeleton Merger: an Unsupervised Aligned Keypoint Detector(骨架合并：无监督的对准关键点检测器)<br>
[paper](https://arxiv.org/pdf/2103.10814.pdf) | [code](https://github.com/eliphatfs/SkeletonMerger)<br><br>

[13] Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding(使用缺失区域编码的循环变换完成不成对的点云)<br>
[paper](https://arxiv.org/abs/2103.07838)<br><br>

[12] Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion(通过双边扩充和自适应融合对实点云场景进行语义分割)<br>
[paper](https://arxiv.org/abs/2103.07074)<br><br>

[11] How Privacy-Preserving are Line Clouds? Recovering Scene Details from 3D Lines(线云如何保护隐私？ 从3D线中恢复场景详细信息)<br>
[paper](https://arxiv.org/pdf/2103.05086.pdf) | [code](https://github.com/kunalchelani/Line2Point)<br><br>

[10] PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency(使用深度空间一致性进行稳健的点云配准)<br>
[paper](https://arxiv.org/abs/2103.05465) | [code](https://github.com/XuyangBai/PointDSC)<br><br>

[9] Robust Point Cloud Registration Framework Based on Deep Graph Matching(基于深度图匹配的鲁棒点云配准框架)<br>
[paper](https://arxiv.org/pdf/2103.04256.pdf) | [code](https://github.com/fukexue/RGM)<br><br>

[8] TPCN: Temporal Point Cloud Networks for Motion Forecasting(面向运动预测的时态点云网络)
- [paper](https://arxiv.org/abs/2103.03067)

[7] PointGuard: Provably Robust 3D Point Cloud Classification(可证明稳健的三维点云分类)
- [paper](https://arxiv.org/abs/2103.03046)

[6] Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges(走向城市规模3D点云的语义分割：数据集，基准和挑战)
- [paper](https://arxiv.org/abs/2009.03137)
- [code](https://github.com/QingyongHu/SensatUrban)

[5] SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration(SpinNet：学习用于3D点云注册的通用表面描述符)
- [paper](https://t.co/xIAWVGQeB2?amp=1)
- [code](https://github.com/QingyongHu/SpinNet)

[4] MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization(通过3D扫描同步进行多主体分割和运动估计)
- [paper](https://arxiv.org/pdf/2101.06605.pdf)
- [code](https://github.com/huangjh-pub/multibody-sync)

[3] Diffusion Probabilistic Models for 3D Point Cloud Generation(三维点云生成的扩散概率模型)
- [paper](https://arxiv.org/abs/2103.01458)
- [code](https://github.com/luost26/diffusion-point-cloud)

[2] Style-based Point Generator with Adversarial Rendering for Point Cloud Completion(用于点云补全的对抗性渲染基于样式的点生成器)
- [paper](https://arxiv.org/abs/2103.02535)

[1] PREDATOR: Registration of 3D Point Clouds with Low Overlap(预测器：低重叠的3D点云的注册)
- [paper](https://arxiv.org/pdf/2011.13005.pdf)
- [code](https://github.com/ShengyuH/OverlapPredator)
- [project](https://overlappredator.github.io/)

<a name="3DReconstruction"/> 

### 三维重建(3D Reconstruction)

[1] PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers（具有透视作物层的3D姿势的几何感知神经重建）
- [paper](https://arxiv.org/abs/2011.13607)

<br>

<a name="ModelCompression"/> 

## 模型压缩(Model Compression)
[1] Manifold Regularized Dynamic Network Pruning（动态剪枝的过程中考虑样本复杂度与网络复杂度的约束）

[2] Learning Student Networks in the Wild（一种不需要原始训练数据的模型压缩和加速技术）
- [paper](https://arxiv.org/pdf/1904.01186.pdf)
- [code](https://github.com/huawei-noah/DAFL)
- 解读：[华为诺亚方舟实验室提出无需数据网络压缩技术](https://zhuanlan.zhihu.com/p/81277796)

<a name="KnowledgeDistillation"/> 

### 知识蒸馏(Knowledge Distillation)

[1] Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation(通过自学来完善自己：通过自我蒸馏提炼特征)
- [paper](https://arxiv.org/pdf/2103.08273.pdf) 
- [code](https://github.com/MingiJi/FRSKD)

[2] Knowledge Evolution in Neural Networks(神经网络中的知识进化)
- [paper](https://arxiv.org/pdf/2103.05152.pdf) 
- [code](https://github.com/ahmdtaha/knowledge_evolution)

[3] Semantic-aware Knowledge Distillation for Few-Shot Class-Incremental Learning(少班级增量学习的语义感知知识蒸馏)
- [paper](https://arxiv.org/abs/2103.04059)

[4] Teachers Do More Than Teach: Compressing Image-to-Image Models(https://arxiv.org/abs/2103.03467)
- [paper](https://arxiv.org/abs/2103.03467) 
- [code](https://github.com/snap-research/CAT)

[5] General Instance Distillation for Object Detection(通用实例蒸馏技术在目标检测中的应用)
- [paper](https://arxiv.org/abs/2103.02340)

[6] Multiresolution Knowledge Distillation for Anomaly Detection(用于异常检测的多分辨率知识蒸馏)
- [paper](https://arxiv.org/abs/2011.11108)

[7] Distilling Object Detectors via Decoupled Features（前景背景分离的蒸馏技术）

<br>

<a name="NNS"/> 

## 神经网络架构(Neural Network Structure)

[1] Coordinate Attention for Efficient Mobile Network Design(协调注意力以实现高效的移动网络设计)
- [paper](https://arxiv.org/abs/2103.02907)
- Code: [https://github.com/Andrew-Qibin/CoordAttention](https://github.com/Andrew-Qibin/CoordAttention)

[2] Inception Convolution with Efficient Dilation Search
- Paper: https://arxiv.org/abs/2012.13587
- Code: None

[3] Rethinking Channel Dimensions for Efficient Model Design(重新考虑通道尺寸以进行有效的模型设计)
- [paper](https://arxiv.org/abs/2007.00992)
- [code](https://github.com/clovaai/rexnet)

[4] Inverting the Inherence of Convolution for Visual Recognition（颠倒卷积的固有性以进行视觉识别）

[5] RepVGG: Making VGG-style ConvNets Great Again
- [paper](https://arxiv.org/abs/2101.03697)
- [code](https://github.com/megvii-model/RepVGG)
- 解读：[RepVGG：极简架构，SOTA性能，让VGG式模型再次伟大](https://zhuanlan.zhihu.com/p/344324470)

[6] Fast and Accurate Model Scaling(快速准确的模型缩放)
- [paper](https://arxiv.org/abs/2103.06877)

[7] Involution: Inverting the Inherence of Convolution for Visual Recognition(反转卷积的固有性以进行视觉识别)
- [paper](https://arxiv.org/abs/2103.06255) 
- [code](https://github.com/d-li14/involution)
<br>

<a name="att"/> 

## Transformer
[1] Transformer Interpretability Beyond Attention Visualization(注意力可视化之外的Transformer可解释性)
- [paper](https://arxiv.org/pdf/2012.09838.pdf)
- [code](https://github.com/hila-chefer/Transformer-Explainability)

[2] UP-DETR: Unsupervised Pre-training for Object Detection with Transformers
- [paper](https://arxiv.org/pdf/2011.09094.pdf)
- 解读：[无监督预训练检测器](https://www.zhihu.com/question/432321109/answer/1606004872)

[3] Pre-Trained Image Processing Transformer(底层视觉预训练模型)
- [paper](https://arxiv.org/pdf/2012.00364.pdf)

<a name="GNN"/> 

### 图神经网络(GNN)

[2] Quantifying Explainers of Graph Neural Networks in Computational Pathology(计算病理学中图神经网络的量化解释器)
- [paper](https://arxiv.org/pdf/2011.12646.pdf)

[1] Sequential Graph Convolutional Network for Active Learning(主动学习的顺序图卷积网络)
- [paper](https://arxiv.org/pdf/2006.10219.pdf)
<br>

<a name="DataProcessing"/> 

## 数据处理(Data Processing)

<a name="DataAugmentation"/> 

### 数据增广(Data Augmentation)

[1] KeepAugment: A Simple Information-Preserving Data Augmentation(一种简单的保存信息的数据扩充)
- [paper](https://arxiv.org/pdf/2011.11778.pdf)

<a name="BatchNormalization"/> 

### 归一化/正则化(Batch Normalization)

[3] Adaptive Consistency Regularization for Semi-Supervised Transfer Learning(半监督转移学习的自适应一致性正则化)
- [paper](https://arxiv.org/abs/2103.02193)
- [code](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning)

[2] Meta Batch-Instance Normalization for Generalizable Person Re-Identification(通用批处理人员重新标识的元批实例规范化)
- [paper](https://arxiv.org/abs/2011.14670)

[1] Representative Batch Normalization with Feature Calibration（具有特征校准功能的代表性批量归一化）

<a name="ImageClustering"/> 

### 图像聚类(Image Clustering)

[2] Improving Unsupervised Image Clustering With Robust Learning（通过鲁棒学习改善无监督图像聚类）
- [paper](https://arxiv.org/abs/2012.11150)
- [code](https://github.com/deu30303/RUC)

[1] Reconsidering Representation Alignment for Multi-view Clustering(重新考虑多视图聚类的表示对齐方式)


<a name="ModelEvaluation"/> 

## 模型评估(Model Evaluation)
[1] Are Labels Necessary for Classifier Accuracy Evaluation?(测试集没有标签，我们可以拿来测试模型吗？)
- [paper](https://arxiv.org/abs/2007.02915)
- [解读](https://zhuanlan.zhihu.com/p/328686799)

<a name="Database"/> 

## 数据集(Database)
[2] Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges(走向城市规模3D点云的语义分割：数据集，基准和挑战)
- [paper](https://arxiv.org/abs/2009.03137)
- [code](https://github.com/QingyongHu/SensatUrban)

[1] Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels（重新标记ImageNet：从单标签到多标签，从全局标签到本地标签）
- [paper](https://arxiv.org/abs/2101.05022)
- [code](https://github.com/naver-ai/relabel_imagenet)


<a name="ActiveLearning"/> 

## 主动学习(Active Learning)

[3] Vab-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning
- [paper](https://github.com/yuantn/MIAL/raw/master/paper.pdf)
- [code](https://github.com/yuantn/MIAL)

[2] Multiple Instance Active Learning for Object Detection（用于对象检测的多实例主动学习）
- [paper](https://github.com/yuantn/MIAL/raw/master/paper.pdf)
- [code](https://github.com/yuantn/MIAL)

[1] Sequential Graph Convolutional Network for Active Learning(主动学习的顺序图卷积网络)
- [paper](https://arxiv.org/pdf/2006.10219.pdf)

<br>
<a name="Few-shotLearning"/> 

## 小样本学习(Few-shot Learning)/零样本
[6] Goal-Oriented Gaze Estimation for Zero-Shot Learning(零样本学习的目标导向注视估计)
- [paper](https://arxiv.org/abs/2103.03433) 
- [code](https://github.com/osierboy/GEM-ZSL)

[5] Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?
- [paper](https://arxiv.org/abs/2012.06166)
- [code](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation)

[4] Counterfactual Zero-Shot and Open-Set Visual Recognition(反事实零射和开集视觉识别)
- [paper](https://arxiv.org/abs/2103.00887)
- [code](https://github.com/yue-zhongqi/gcm-cf)

[3] Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection(小样本目标检测的语义关系推理)
- [paper](https://arxiv.org/abs/2103.01903)

[2] Few-shot Open-set Recognition by Transformation Consistency(转换一致性很少的开放集识别)

[1] Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning(探索少量学习的不变表示形式和等变表示形式的互补强度)
- [paper](https://arxiv.org/abs/2103.01315)


<a name="ContinualLearning"/> 

## 持续学习(Continual Learning/Life-long Learning)

[2] Rainbow Memory: Continual Learning with a Memory of Diverse Samples（不断学习与多样本的记忆）

[1] Learning the Superpixel in a Non-iterative and Lifelong Manner(以非迭代和终身的方式学习超像素)
<br>

<a name="VisualReasoning"/> 

## 视觉推理(Visual Reasoning)

[1] Transformation Driven Visual Reasoning(转型驱动的视觉推理)
- [paper](https://arxiv.org/pdf/2011.13160.pdf)
- [code](https://github.com/hughplay/TVR)
- [project](https://hongxin2019.github.io/TVR/)




<a name="domain"/> 

## 迁移学习/domain/自适应](#domain)

[4] Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning(通过域随机化和元学习对视觉表示进行连续调整)
- [paper](https://arxiv.org/abs/2012.04324)

[3] Domain Generalization via Inference-time Label-Preserving Target Projections(基于推理时间保标目标投影的区域泛化)
- [paper](https://arxiv.org/abs/2103.01134)

[2] MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive  Sensing(可伸缩的自适应视频压缩传感重建)
- [paper](https://arxiv.org/abs/2103.01786)
- [code](https://github.com/xyvirtualgroup/MetaSCI-CVPR2021)

[1] FSDR: Frequency Space Domain Randomization for Domain Generalization(用于域推广的频域随机化)
- [paper](https://arxiv.org/abs/2103.02370)



<a name="ContrastiveLearning)"/> 

## 对比学习(Contrastive Learning)

[1] Fine-grained Angular Contrastive Learning with Coarse Labels(粗标签的细粒度角度对比学习)
- [paper](https://arxiv.org/abs/2012.03515)


<a name="ImageRetrieval"/> 

## 图像视频检索(Image Retrieval)

[1] QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval(实用的查询高效的图像检索黑盒攻击)
- [paper](https://arxiv.org/abs/2103.02927)



<a name="100"/> 

## 暂无分类

Learning Asynchronous and Sparse Human-Object Interaction in Videos(视频中异步稀疏人-物交互的学习)
- [paper](https://arxiv.org/abs/2103.02758)

Self-supervised Geometric Perception(自我监督的几何知觉)
- [paper](https://arxiv.org/abs/2103.03114)

Quantifying Explainers of Graph Neural Networks in Computational Pathology(计算病理学中图神经网络的量化解释器)
- [paper](https://arxiv.org/pdf/2011.12646.pdf)

Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts(探索具有对比场景上下文的数据高效3D场景理解)
- [paper](http://arxiv.org/abs/2012.09165)
- [project](http://sekunde.github.io/project_efficient)
- [video](http://youtu.be/E70xToZLgs4)

Data-Free Model Extraction(无数据模型提取)
- [paper](https://arxiv.org/abs/2011.14779)

Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition(用于【位置识别】的局部全局描述符的【多尺度融合】)
- [paper](https://arxiv.org/pdf/2103.01486.pdf)
- [code](https://github.com/QVPR/Patch-NetVLAD)

Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting with their Explanations(适用于正确概念的权利：通过可解释性来修正神经符号概念)
- [paper](https://arxiv.org/abs/2011.12854)

Multi-Objective Interpolation Training for Robustness to Label Noise(多目标插值训练的鲁棒性)
- [paper](https://arxiv.org/abs/2012.04462)
- [code](https://git.io/JI40X)

VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs(【文本生成】VX2TEXT：基于视频的文本生成的端到端学习来自多模式输入)
- [paper](https://arxiv.org/pdf/2101.12059.pdf)

Scan2Cap: Context-aware Dense Captioning in RGB-D Scans(【图像字幕】Scan2Cap：RGB-D扫描中的上下文感知密集字幕)
- [paper](https://arxiv.org/abs/2012.02206)
- [code](https://github.com/daveredrum/Scan2Cap)
- [project](https://daveredrum.github.io/Scan2Cap/)
- [video](https://youtu.be/AgmIpDbwTCY)

Hierarchical and Partially Observable Goal-driven Policy Learning with  Goals Relational Graph(基于目标关系图的分层部分可观测目标驱动策略学习)
- [paper](https://arxiv.org/abs/2103.01350)

ID-Unet: Iterative Soft and Hard Deformation for View Synthesis(视图合成的迭代软硬变形)
- [paper](https://arxiv.org/abs/2103.02264)

PML: Progressive Margin Loss for Long-tailed Age Classification(【长尾分布】【图像分类】长尾年龄分类的累进边际损失)
- [paper](https://arxiv.org/abs/2103.02140)

Diversifying Sample Generation for Data-Free Quantization（【图像生成】多样化的样本生成，实现无数据量化）
- [paper](https://arxiv.org/abs/2103.01049)

Domain Generalization via Inference-time Label-Preserving Target Projections（通过保留推理时间的目标投影进行域泛化）
- [paper](https://arxiv.org/pdf/2103.01134.pdf)

DeRF: Decomposed Radiance Fields（分解的辐射场）
- [project](https://ubc-vision.github.io/derf/)

Densely connected multidilated convolutional networks for dense prediction tasks（【密集预测】密集连接的多重卷积网络，用于密集的预测任务）
- [paper](https://arxiv.org/abs/2011.11844)

VirTex: Learning Visual Representations from Textual Annotations（【表示学习】从文本注释中学习视觉表示）
- [paper](https://arxiv.org/abs/2006.06666)
- [code](https://github.com/kdexd/virtex)

Weakly-supervised Grounded Visual Question Answering using Capsules（使用胶囊进行弱监督的地面视觉问答）

FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation（【视频插帧】FLAVR：用于快速帧插值的与流无关的视频表示）
- [paper](https://arxiv.org/pdf/2012.08512.pdf)
- [code](https://tarun005.github.io/FLAVR/Code)
- [project](https://tarun005.github.io/FLAVR/)

Probabilistic Embeddings for Cross-Modal Retrieval（跨模态检索的概率嵌入）
- [paper](https://arxiv.org/abs/2101.05068)

Self-supervised Simultaneous Multi-Step Prediction of Road Dynamics and Cost Map(道路动力学和成本图的自监督式多步同时预测)

IIRC: Incremental Implicitly-Refined Classification(增量式隐式定义的分类)
- [paper](https://arxiv.org/abs/2012.12477)
- [project](https://chandar-lab.github.io/IIRC/)

Fair Attribute Classification through Latent Space De-biasing(通过潜在空间去偏的公平属性分类)
- [paper](https://arxiv.org/abs/2012.01469)
- [code](https://github.com/princetonvisualai/gan-debiasing)
- [project](https://princetonvisualai.github.io/gan-debiasing/)

Information-Theoretic Segmentation by Inpainting Error Maximization(修复误差最大化的信息理论分割)
- [paper](https://arxiv.org/abs/2012.07287)

UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pretraining(【视频语言学习】UC2：通用跨语言跨模态视觉和语言预培训)

Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling(通过稀疏采样进行视频和语言学习)
- [paper](https://arxiv.org/pdf/2102.06183.pdf)
- [code](https://github.com/jayleicn/ClipBERT)

D-NeRF: Neural Radiance Fields for Dynamic Scenes(D-NeRF：动态场景的神经辐射场)
- [paper](https://arxiv.org/abs/2011.13961)
- [project](https://www.albertpumarola.com/research/D-NeRF/index.html)

Weakly Supervised Learning of Rigid 3D Scene Flow(刚性3D场景流的弱监督学习)
- [paper](https://arxiv.org/pdf/2102.08945.pdf)
- [code](https://arxiv.org/pdf/2102.08945.pdf)
- [project](https://3dsceneflow.github.io/)

<br>

<a name="2"/> 

## CVPR2021 Oral

[23] Self-supervised Geometric Perception(自我监督的几何知觉)
- [paper](https://arxiv.org/abs/2103.03114)

[22] DeepTag: An Unsupervised Deep Learning Method for Motion Tracking on  Cardiac Tagging Magnetic Resonance Images(一种心脏标记磁共振图像运动跟踪的无监督深度学习方法)
- [paper](https://arxiv.org/abs/2103.02772)

[21] Modeling Multi-Label Action Dependencies for Temporal Action Localization(为时间动作本地化建模多标签动作相关性)
- [paper](https://arxiv.org/pdf/2103.03027.pdf)

[20] HPS: localizing and tracking people in large 3D scenes from wearable sensors(通过可穿戴式传感器对大型3D场景中的人进行定位和跟踪)

[19] Real-Time High Resolution Background Matting(实时高分辨率背景抠像)
- [paper](https://arxiv.org/abs/2012.07810)
- [code](https://github.com/PeterL1n/BackgroundMattingV2)
- [project](https://grail.cs.washington.edu/projects/background-matting-v2/)
- [video](https://youtu.be/oMfPTeYDF9g)

[18] Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts(探索具有对比场景上下文的数据高效3D场景理解)
- [paper](http://arxiv.org/abs/2012.09165)
- [project](http://sekunde.github.io/project_efficient)
- [video](http://youtu.be/E70xToZLgs4)

[17] Robust Neural Routing Through Space Partitions for Camera Relocalization in Dynamic Indoor Environments(在动态室内环境中，通过空间划分的鲁棒神经路由可实现摄像机的重新定位)
- [paper](https://arxiv.org/abs/2012.04746)
- [project](https://ai.stanford.edu/~hewang/)

[16] MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization(通过3D扫描同步进行多主体分割和运动估计)
- [paper](https://arxiv.org/pdf/2101.06605.pdf)
- [code](https://github.com/huangjh-pub/multibody-sync)

[15] Categorical Depth Distribution Network for Monocular 3D Object Detection(用于单目三维目标检测的分类深度分布网络)
- [paper](https://arxiv.org/abs/2103.01100)

[14] PatchmatchNet: Learned Multi-View Patchmatch Stereo(学习多视图立体声)
- [paper](https://arxiv.org/abs/2012.01411)
- [code](https://github.com/FangjinhuaWang/PatchmatchNet)

[13] Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning(通过域随机化和元学习对视觉表示进行连续调整)
- [paper](https://arxiv.org/abs/2012.04324)

[12] Single-Stage Instance Shadow Detection with Bidirectional Relation Learning(具有双向关系学习的单阶段实例阴影检测)

[11] Neural Geometric Level of Detail:Real-time Rendering with Implicit 3D Surfaces(神经几何细节水平：隐式3D曲面的实时渲染)
- [paper](https://arxiv.org/abs/2101.10994)
- [code](https://github.com/nv-tlabs/nglod)
- [project](https://nv-tlabs.github.io/nglod/)

[9] PREDATOR: Registration of 3D Point Clouds with Low Overlap(预测器：低重叠的3D点云的注册)
- [paper](https://arxiv.org/pdf/2011.13005.pdf)
- [code](https://github.com/ShengyuH/OverlapPredator)
- [project](https://overlappredator.github.io/)

[8] Domain Generalization via Inference-time Label-Preserving Target Projections(通过保留推理时间的目标投影进行域泛化)
- [paper](https://arxiv.org/abs/2103.01134)

[7] Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction(全局一致的非刚性重建的神经变形图)
- [paper](https://arxiv.org/abs/2012.01451)
- [project](https://aljazbozic.github.io/neural_deformation_graphs/)
- [video](https://youtu.be/vyq36eFkdWo)

[6] Fine-grained Angular Contrastive Learning with Coarse Labels(粗标签的细粒度角度对比学习)
- [paper](https://arxiv.org/abs/2012.03515)

[5] Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling(通过稀疏采样进行视频和语言学习)
- [paper](https://arxiv.org/pdf/2102.06183.pdf)
- [code](https://github.com/jayleicn/ClipBERT)

[4] Cross-View Regularization for Domain Adaptive Panoptic Segmentation(用于域自适应全景分割的跨视图正则化)
- [paper](https://arxiv.org/abs/2103.02584)

[3] Image-to-image Translation via Hierarchical Style Disentanglement(通过分层样式分解实现图像到图像的翻译)
- [paper](https://arxiv.org/abs/2103.01456)
- [code](https://github.com/imlixinyang/HiSD)

[2] Towards Open World Object Detection(开放世界中的目标检测)
- [paper](Towards Open World Object Detection)
- [code](https://github.com/JosephKJ/OWOD)

[1] End-to-End Video Instance Segmentation with Transformers(使用Transformer的端到端视频实例分割) 
- [paper](https://arxiv.org/abs/2011.14503)


<br>

<a name="300"/> 

## CVPR2021的论文解读

* [CVPR 2021 | GFLV2：目标检测良心技术，无Cost涨点!](https://zhuanlan.zhihu.com/p/313684358)
* [CVPR 2021 | 上交和国科大提出DCL：旋转目标检测新方法](https://zhuanlan.zhihu.com/p/354373013)
* [CVPR 2021 | 涨点神器！IC-Conv：使用高效空洞搜索的Inception卷积，全方位提升！]()
* [CVPR 2021 Oral | 层次风格解耦：人脸多属性篡改终于可控了！](https://zhuanlan.zhihu.com/p/354258056)
* [CVPR 2021 | Transformer进军low-level视觉！北大华为等提出预训练模型IPT]()

<br>

