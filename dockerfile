FROM ubuntu

FROM python:3.6

RUN apt-get update && apt-get install -y unzip && pip install keras==2.1.6 tensorflow-cpu==1.15.0 h5py==2.10.0 image python-dateutil pandas numpy

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /wd

#to COPY the remote file at working directory in container
COPY . ./

# As the algorithm will be executed by a non-root user, the output folder has to be writable for all
RUN mkdir temp -p && mkdir /data/logs -p && mkdir -p /data/output && chmod a+rx /data && chmod a+rwx /data/output

# The startAlgorithm script is the entry point of the container. It is the script that will be executed 
# when the container is run by SYLVA IT infrastructure. The script itself should just start your algorithm 
# to get all files in folder /data/input processed.
COPY src/bin/startAlgorithm /bin/startAlgorithm
RUN chmod a+x /bin/startAlgorithm

RUN chmod a+x src/algorithm.sh

RUN chmod a+rwx temp


#CMD "src/algorithm.sh"