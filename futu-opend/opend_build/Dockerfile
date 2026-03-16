# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 1. 安装基础工具
RUN apt-get update && \
    apt-get install -y wget ca-certificates openssl libssl-dev libffi-dev net-tools locales && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

# 2. 智能下载并解压
# 逻辑说明：下载 -> 解压 -> 找到 FutuOpenD 文件所在的真实目录 -> 把该目录下的所有文件移动到 /app -> 清理垃圾
RUN wget -O opend.tar.gz "https://www.futunn.com/download/fetch-lasted-link?name=opend-ubuntu" && \
    tar -xf opend.tar.gz && \
    # 关键修改：使用 find 命令查找 FutuOpenD 文件所在的路径，不管它藏在第几层目录里
    REAL_DIR=$(find . -type f -name "FutuOpenD" -print -quit | xargs dirname) && \
    echo "Found OpenD in directory: $REAL_DIR" && \
    mv "$REAL_DIR"/* . && \
    rm -rf Futu_OpenD* opend.tar.gz

# 3. 赋予权限
RUN chmod +x FutuOpenD

# 4. 暴露端口
EXPOSE 11111

# 5. 启动
CMD ["./FutuOpenD", "-xml_path=./FutuOpenD.xml"]
