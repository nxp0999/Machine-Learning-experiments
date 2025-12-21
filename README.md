# CS 6375 Machine Learning Project 3 - Deep Learning


## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements Check (requirements.txt)
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Project Structure

````
project3.zip
├── CS6375_Project3_Report.pdf              (Project report)
├── AI Transcript.pdf                       (AI Transcript)
├── README.md                          
├── requirements.txt                     
├── results.txt                             (Summary of results)
├── src/                                    (Source code)
    ├── mlp.py                              (Code for Part 1)
    ├── mnist.py                            (Code for Part 2 - MNIST CNNs)
    ├── cifar10.py                          (Code for Part 2 - CIFAR-10 CNNs)
    ├── mnist_mlp_results.json              (Experimental results)
    ├── cifar10_mlp_results.json
    ├── mnist_cnn_results.json
    └── cifar10_cnn_results.json
````

## Running the Code

For running all the scripts together : 
   python main.py

For implementing each script separately: 
1. MLP Experiments :
   python mlp.py
   - Runs all MLP experiments on MNIST and CIFAR-10
   - Outputs: mnist_mlp_results.json, cifar10_mlp_results.json
   - Runtime: ~1-1.5 hours on T4 GPU

2. MNIST CNN Experiments (Part 1):
   python mnist_cnn.py
   - Runs all CNN experiments on MNIST
   - Outputs: mnist_cnn_results.json
   - Runtime: ~30-45 minutes on T4 GPU

3. CIFAR-10 CNN Experiments (Part 2):
   python cifar10_cnn.py
   - Runs all CNN experiments on CIFAR-10
   - Outputs: cifar10_cnn_results.json
   - Runtime: ~1.5-2 hours on T4 GPU


## Reproducibility
- Random seed: 42
- Device: CUDA GPU (Google Colab T4)
- Framework: PyTorch



