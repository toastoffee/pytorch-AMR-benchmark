# Train results

## 1. Baseline

|   Model    | Top-1 ACC(%) | Top-5 ACC(%) |
|:----------:|:------------:|:------------:|
| ResNet-18  |    58.4%     |    86.9%     |
| ResNet-34  |    56.4%     |    86.7%     |
| ResNet-50  |    56.1%     |    86.6%     | 
| ResNet-101 |    56.9%     |    86.3%     |
| ResNet-152 |    56.9%     |    86.7%     |


## 2. sim-KD

| Student Model | Teacher Model | Top-1 ACC(%) | Top-5 ACC(%) |
|:--------------|:-------------:|:------------:|:------------:|
| ResNet-18     |   ResNet-34   |    57.1%     |    86.5%     |
| ResNet-18     |   ResNet-50   |    57.1%     |    86.4%     | 
| ResNet-18     |  ResNet-101   |    57.8%     |    86.8%     |
| ResNet-18     |  ResNet-152   |    57.9%     |    86.8%     |

| Student Model | Teacher Model | Top-1 ACC(%) | Top-5 ACC(%) |
|:--------------|:-------------:|:------------:|:------------:|
| CNN2          |   ResNet-34   |    57.8%     |    86.9%     |
| CNN2          |   ResNet-50   |    57.9%     |    86.9%     | 
| CNN2          |  ResNet-101   |    57.8%     |    86.8%     |
