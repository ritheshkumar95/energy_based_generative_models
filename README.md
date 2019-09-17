# Maximum Entropy Generators for Energy-Based Models

All experiments have tensorboard visualizations for samples / density / train curves etc.

1. To run the toy data experiments:
```
python scripts/train/ebm_toy.py --dataset swissroll --save_path logs/swissroll
```

2. To run the discrete mode collapse experiment:
```
python scripts/train/ebm_mnist.py --save_path logs/mnist_3 --n_stack 3
```

This requires the pretrained mnist classifier:
```
python scripts/train/mnist_classifier.py
```

3. To run the CIFAR image generation experiment:
```
python scripts/train/ebm_cifar.py --save_path logs/cifar
```

To run the MCMC evalulations on CIFAR data:
```
python scripts/test/eval_metrics_cifar --load_path logs/cifar --n_samples 50000 --mcmc_iters 5 --temp .01
```

NOTE: This requires cloning the TTUR repo in the current working directory (https://github.com/bioinf-jku/TTUR).

4. To run the CelebA image generation experiments:
```
python scripts/train/ebm_celeba.py --save_path logs/celeba
```

NOTE: Results are subject to PyTorch version. I have already noticed variance in quantitative numbers with PyTorch version upgrades.