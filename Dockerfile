FROM ubuntu

USER root
RUN ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
ENV DLMEETUP /opt/DLMeetup
WORKDIR $DLMEETUP

RUN sed s/archive\.ubuntu\.com/se\.archive\.ubuntu\.com/g /etc/apt/sources.list > /tmp/sources.list && cp /tmp/sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install --assume-yes \
    ntp \
    python \
    python-pip \
    ranger \
    ipython \
    htop \
    ssh \
    vim

COPY apt-build-dep-requirements.txt /tmp/apt-build-dep-requirements.txt
RUN apt-get build-dep --assume-yes\
    $(cat /tmp/apt-build-dep-requirements.txt | tr '\n' ' ')

COPY apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get update && apt-get install --assume-yes \
    $(cat /tmp/apt-requirements.txt | tr '\n' ' ')

COPY requirements.txt /tmp/requirements.txt
RUN pip install --requirement /tmp/requirements.txt
RUN pip install nose

COPY . .

CMD [ "nosetests", "-sv", "/bin/bash"]
