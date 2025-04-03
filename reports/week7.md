### Week 7 Report

#### 本周工作

- 进行实验，确认`context_manager.py`的`ContextManager`中使用`MemoryUnit`进行显存卸载到内存中以及从内存中加载比较耗时。这也符合预期

- 由于实验目的是在llama.cpp上进行优化。所以本周进行了llama.cpp的环境配置和运行测试。

    - llama.cpp现在需要cmake进行编译，所以需要手动编译cmake

        ```
        wget https://github.com/Kitware/CMake/releases/download/v4.0.0/cmake-4.0.0.tar.gz
        tar axf cmake-4.0.0.tar.gz
        cd cmake-4.0.0
        ./bootstrap
        ./configure --prefix={我的安装目录}/cmake
        make
        make install

        export PATH={我的安装目录}/cmake/bin:$PATH
        ```

        但是在上面安装中遇到了缺少OpenSSL的报错，所以需要自行编译OpenSSL，然后设置OPENSSL_ROOT_DIR环境变量后解决

        ```
        wget https://github.com/openssl/openssl/releases/download/openssl-3.4.1/openssl-3.4.1.tar.gz
        tar axf openssl-3.4.1.tar.gz
        cd openssl-3.4.1
        ./config --prefix={我的安装目录}/openssl
        make
        make install

        export OPENSSL_ROOT_DIR={我的安装目录}/openssl
        ```

    - 参考https://qwen.readthedocs.io/zh-cn/latest/run_locally/llama.cpp.html，编译并尝试在llama.cpp上运行一个模型

        上述文档中使用make进行编译，但是现在llama.cpp已经改为cmake编译

        编译llama.cpp的CUDA版本：

        ```
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release
        ```

        编译过程中发生了`gcc: error: unrecognized command-line option ‘-compress-mode=size’`报错，查询https://github.com/ggml-org/llama.cpp/issues/12325，发现是显卡驱动的版本新于nvcc编译器导致，手动设置`export CUDACXX=/usr/local/cuda-12.8/bin/nvcc`后解决问题。

        最后完成了编译。

        qwen模型下载中也和这个文档中有所不同，gguf文件拆分成了两部分，将两部分均下载后，修改命令，最后跑通了llama.cpp的编译流程

        ```
        ./llama-cli -m qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
            -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
            -fa -ngl 80 -n 512
        ```

    - 阅读了一些`llama.cpp/src/llama-kv-cache.cpp`等文件中的源码，准备后续的代码修改工作。

#### 下周工作计划

- 阅读llama.cpp的源码，寻找InfLLM中的实现在llama.cpp中的对应位置。尝试在llama.cpp中进行InfLLM的功能复现