### Week 6 Report

#### 本周工作

- 首先找了一些博客文章，学习了KV cache相关技术的基础

- 阅读了InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory（这篇文章通过offload的方式减少了显存占用等方式，提高在长文本上的性能，海里捞针测试中的性能很好），RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval和PQCache: Product Quantization-based KVCachefor Long Context LLM Inference

- 复现InfLLM的结果：

    另外两篇文章似乎没有足够的代码和细节，而InfLLM给出了部分实现细节，所以选择复现InfLLM。

    不过即使如此，似乎InfLLM的环境比较旧，配环境时遇到了出乎意料的多的问题

    - 下面是一些环境配置细节

        使用conda(python=3.12)进行包管理，在RTX 4090, CUDA=12.6上安装pytorch=2.6.0环境

        安装环境和下载数据时，可能需要使用代理以保证网络稳定

        为了测试benchmark，需要将requirements.txt中部分注释掉的包解除注释

    - 复现结果时，出现了如下bug

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 9, in <module>
            from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
        ModuleNotFoundError: No module named 'inf_llm'
        ```

        加入PYTHONPATH=.的环境变量后解决

    - 然后又出现了huggingface要求登录的提示

        加入HUGGING_FACE_HUB_TOKEN的环境变量后解决

    - 然后出现如下报错：

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 289, in <module>
            model, tokenizer = get_model_and_tokenizer(args.model)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 56, in get_model_and_tokenizer
            model = patch_hf(model, config.type, **config)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/inf_llm/utils/patch.py", line 152, in patch_hf
            hf_rope = model.model.layers[0].self_attn.rotary_emb 
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的家目录>/miniconda3/envs/infllm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1928, in __getattr__
            raise AttributeError(
        AttributeError: 'MistralAttention' object has no attribute 'rotary_emb'
        ```

        猜测是版本不同导致rotary_emb没有了，于是检索到一个类似问题

        结合 [AttributeError: 'LlamaAttention' object has no attribute 'rotary_emb'](https://github.com/unslothai/unsloth/issues/1443) 中的回答，执行如下代码，改变环境配置

        ```bash
        pip install unsloth && pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps https://github.com/unslothai/unsloth/archive/refs/tags/November-2024.zip
        pip uninstall tokenizers
        pip uninstall transformers
        pip install tokenizers==0.20.3 transformers==4.46.1
        ```

        如此操作，解决了如上报错

        

#### 下周工作计划

- 进一步研究InfLLM的细节，进一步理解InfLLM的算法设计，并寻找可以提升之处
- 阅读其他有关KV cache论文，并结合实验复现结果和代码阅读理解论文