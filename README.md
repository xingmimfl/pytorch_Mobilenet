# Mobilenet with PyTorch
**Note:** This project is pytorch implementation of [Mobilenet](https://arxiv.org/abs/1704.04861).

### Performance

Trained on ImageNet, get Prec@1 63.550% Prec@5 85.650%. Unfortunately, due to limitation of GPU resources, training was
stopped at 39 epoch. So I think it may get better results if the training could be completed. During the training, I set batch_size=256, learning_rate=0.1 which decayed every 30 epoch by 10. 


### Training on ImageNet

```bash
python main.py -b 256 $Imagenetdir
```

License: MIT license (MIT)
