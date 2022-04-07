ARG TORCH=1.10.0
ARG CUDA=11.3
ARG CUDNN=8

FROM pytorch/pytorch:${TORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN pip install tensorboard==2.6.0 \
    facenet-pytorch==2.5.2 \
    scikit-image==0.19.2 \
    pyyaml==6.0

ARG UID
ARG GID
ARG UNAME

RUN groupadd -g ${GID} ${UNAME} && \
    useradd -u ${UID} -g ${UNAME} -m ${UNAME}

ENV UID=${UID} \
    GID=${GID} \
    UNAME=${UNAME} \
    LANG=C.UTF-8 \
    HOME=/home/${UNAME}

WORKDIR /home/${UNAME}
USER ${UNAME}
