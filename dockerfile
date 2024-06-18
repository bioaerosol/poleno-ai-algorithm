FROM ubuntu

FROM python:3.6

RUN apt-get update && apt-get install -y unzip && pip install keras==2.1.6 tensorflow-cpu==1.15.0 h5py==2.10.0 image python-dateutil pandas numpy

#Labels as key value pair
LABEL Maintainer="SYLVA Poleno"

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /wd

#to COPY the remote file at working directory in container
COPY . ./

RUN mkdir temp -p

COPY src/bin/startAlgorithm /bin/startAlgorithm
RUN chmod a+x /bin/startAlgorithm

CMD "src/algorithm.sh"