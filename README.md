# Mobilenet with PyTorch
**Note:** This project is pytorch implementation of [Mobilenet](https://arxiv.org/abs/1704.04861).

### Performance

Trained on ImageNet, get Prec@1 63.550% Prec@5 85.650%. Unfortunately, due to limitation of GPU resources, training was
stopped at 39 epoch. So I think it may get better results if the training could be completed. During the training, I set batch_size=256, learning_rate=0.1 which decayed every 30 epoch by 10. 


### Training on ImageNet

```bash
python main.py -b 256 $Imagenetdir
```

### Training mobilenet v3 on ImageNet

```bash
CUDA_VISIBLE_DEVICES=2,3 python main.py -b 256 $Imagenetdir
```
|commit version| Acc@1 | Acc@5
------------------------------
|d98433a | 56.852| 79.990

License: MIT license (MIT)
