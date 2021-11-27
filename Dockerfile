FROM tensorflow/tensorflow:2.6.0rc1
MAINTAINER chenruifeng<ruifeng.chen.cn@gmail.com>

COPY ./psgdworker ./worker/

RUN pip install pandas

EXPOSE 15387

CMD python ./worker/worker.py
