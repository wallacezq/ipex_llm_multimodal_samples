# Multimodal LLM with IPEX-LLM

The repo provides a few examples of running large generative multimodal models using IPEX-LLM

>  [!Note] 
> 
> The examples uses datasets / videos from public domain. See [PKU-YuanGroup/Video-LLaVA (github.com)](https://github.com/PKU-YuanGroup/Video-LLaVA/tree/main/videollava/serve/examples) and [Intel/Video_Summarization_For_Retail](https://huggingface.co/datasets/Intel/Video_Summarization_For_Retail)



## Recommended System

- Meteor Lake based NUC (preferably with Core Intel Ultra 7 processor)

- Minimum 64GB DDR5 memory

- 1TB NVMe

- OS: Windows* 11 / Linux* 22.04 or later

## Environment Setup

- Install IPEX-LLM. Follow the guide [here](https://github.com/intel-analytics/ipex-llm/blob/main/README.md#install).
- Install all python packages dependencies, as below:
  
  ```bash
  $ cd ipex_llm_multimodal_samples
  $ pip install -r requirements
  ```
  
  

## Run the example

```bash
$ cd ipex_llm_multimodal_samples
$ python video-llava.py
```

> [!Note]
> 
> The script can take a bit longer (~3-5mins) for the gradio server to be up and running due to loading and warm-up of the generative model. So please be patient. :smile:


