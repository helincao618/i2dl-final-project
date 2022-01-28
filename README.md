## Final Project of Introduction to Deep Learning (EI78055) held by TUM-HCR

An PyTorch Reimplementation of PointNet++

### Requirements
This implementation uses Python 3.6, Pytorch1.4.0, cudatoolkit 10.0. We recommend to use conda to deploy the environment.

        ```
	conda create -n i2dl python=3.6
	
	conda activate i2dl
	
	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
	
	pip install fire tqdm
        ```

## Classification
- **Start**
    - Dataset: [ModelNet40](https://modelnet.cs.princeton.edu/), download it from [Official Site](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
    - Train
        ```
        python train_clss.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_clss.py --data_root ./modelnet40_normal_resampled --log_dir cls
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_cls pointnet2_cls_ssg data_root checkpoint num_points
        
        eg.
        python evaluate.py evaluate_cls pointnet2_cls_ssg ./modelnet40_normal_resampled \
        ./cls/checkpoints/pointnet2_cls_250.pth 1024
        ``` 

## Part Segmentation
- **Start**
    - Dataset: [ShapeNet part](https://shapenet.cs.stanford.edu/iccv17/#dataset), download it from [Official Site](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
    - Train
        ```
        python train_part_seg.py --data_root your_data_root --log_dir your_log_dir

        eg.
        python train_part_seg.py --data_root ./shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --log_dir seg --batch_size 64
        ```
    - Evaluate
    
        ```
        python evaluate.py evaluate_seg data_root checkpoint
        
        eg.
        python evaluate.py evaluate_seg ./shapenetcore_partanno_segmentation_benchmark_v0_normal \
        ./seg/checkpoints/pointnet2_cls_250.pth
        ```
	
## Reference

- [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
