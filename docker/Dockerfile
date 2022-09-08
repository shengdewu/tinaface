FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

COPY requirements /home/vedadet/requirements
COPY vedacore  /home/vedadet/vedacore
COPY vedadet   /home/vedadet/vedadet
COPY setup.py /home/vedadet

RUN pip3 install -r /home/vedadet/requirements/runtime.txt
RUN cd /home/vedadet && pip install -v -e .

COPY tools/trainval.py /home/vedadet

WORKDIR /home/vedadet
ENTRYPOINT ["python3", "trainval.py"]