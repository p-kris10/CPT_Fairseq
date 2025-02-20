
# **Quantization and Cyclic Precision Training **  

This repository explores post-training quantization and quantization-aware fine-tuning for the RoBERTa model using Fairseq. The project integrates techniques inspired by [Cyclic Precision Training (CPT)](https://arxiv.org/pdf/2101.09868) to improve the efficiency of large language models while maintaining performance on downstream tasks.  

## **Overview**  

With the growing demand for deploying transformer-based models on resource-constrained devices, quantization has emerged as a key technique to reduce model size and inference latency. This project focuses on:  

- Implementing post-training quantization for RoBERTa on the GLUE benchmark  
- Integrating cyclic precision training into the fine-tuning process  
- Evaluating the impact of different quantization strategies on model accuracy  

## **Key Features**  

✅ **Post-Training Quantization (PTQ)**: Applies weight and activation quantization to a pretrained RoBERTa model, calibrating the min/max values using a moving average strategy.  
✅ **Cyclic Precision Training (CPT) Integration**: Dynamically varies weight precision during training to enhance robustness to quantization.  
✅ **Flexible Configuration**: Supports various weight precisions (4-bit, 8-bit, 16-bit) and allows fine-tuning with frozen or trainable backbone layers.  
✅ **Performance Analysis**: Measures accuracy degradation across different quantization settings and compares sensitivity between self-attention and feed-forward layers.  

## **Implementation Details**  

- **Model Modifications**:  
  - Quantized transformer layers implemented in `fairseq/modules/transformer_layer.py` and `multihead_attention.py`  
  - Custom quantization logic in `quantize.py`  
  - Training state control in `fairseq/modules/state.py`  

- **Running Post-Training Quantization (PTQ)**:  
  Set `CYCLIC=False` in `state.py`, specify the desired bit precision (`BITS=4` or `BITS=8`), then execute the training command.  

- **Running Quantization-Aware Fine-Tuning (QAT) with CPT**:  
  Set `CYCLIC=True` and configure the desired precision cycling strategy before running:  
  ```bash
  CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name rte +task.data=../../../RTE-bin checkpoint.restore_file=/path/to/pretrained/model.pt
  ```

## **Experiments and Insights**  

- Compared accuracy drop between full-precision and quantized models  
- Analyzed quantization sensitivity across self-attention and feed-forward layers  
- Evaluated various cyclic precision strategies (4-bit ↔ 8-bit, 8-bit ↔ 16-bit, etc.)  
- Benchmarked against standard fine-tuning approaches  

## **Future Directions**  

🔹 Extending the approach to other transformer architectures like BERT and T5  
🔹 Exploring mixed-precision training for improved trade-offs  
🔹 Optimizing inference speed and memory efficiency for edge deployment  

