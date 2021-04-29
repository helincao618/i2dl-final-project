## Introduction

An PyTorch Reimplementation of PointNet++

### Requirements

- PyTorch, Python3, TensorboardX, tqdm, fire

## Classification
- **Start**
    - Dataset: [ModelNet40](https://modelnet.cs.princeton.edu/), download it from [Official Site](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
    - Train
        ```
        python train_clss.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_clss.py --data_root /root/modelnet40_normal_resampled --log_dir cls_ssg_1024
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_cls model data_root checkpoint npoints
        
        eg.
        python evaluate.py evaluate_cls pointnet2_cls_ssg  /root/modelnet40_normal_resampled \
        checkpoints/pointnet2_cls_250.pth 1024
        
        python evaluate.py evaluate_cls pointnet2_cls_msg root/modelnet40_normal_resampled \
        checkpoints/pointnet2_cls_250.pth 1024
        ``` 

    - **Start to train**
        ```
        python train_custom_cls.py --data_root your_datapath/CustomData --nclasses 2 --npoints 2048
        ```
    - **Start to evaluate**
        ```
        python evaluate_custom.py evaluate_cls pointnet2_cls_ssg your_datapath/CustomData work_dirs/checkpoints/pointnet2_cls_250.pth 2
        ```

## Part Segmentation
- **Start**
    - Dataset: [ShapeNet part](https://shapenet.cs.stanford.edu/iccv17/#dataset), download it from [Official Site](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
    - Train
        ```
        python train_part_seg.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_part_seg.py --data_root /root/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --log_dir seg_ssg --batch_size 64
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_seg data_root checkpoint
        
        eg.
        python evaluate.py evaluate_seg /root/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        seg_ssg/checkpoints/pointnet2_cls_250.pth
        ```
	
## Reference

- [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
