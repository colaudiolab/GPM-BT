#! /bin/env bash

mkdir -p data
python /media/dxy/Lexar/beans-main/scripts/watkins.py
python /media/dxy/Lexar/beans-main/scripts/bats.py
python /media/dxy/Lexar/beans-main/scripts/cbi.py
python /media/dxy/Lexar/beans-main/scripts/humbugdb.py
python /media/dxy/Lexar/beans-main/scripts/dogs.py
python /media/dxy/Lexar/beans-main/scripts/dcase.py
python /media/dxy/Lexar/beans-main/scripts/enabirds.py
mkdir data/hiceas
wget https://storage.googleapis.com/ml-bioacoustics-datasets/hiceas_1-20_minke-detection.zip -O data/hiceas/hiceas.zip
unzip data/hiceas/hiceas.zip -d data/hiceas
python /media/dxy/Lexar/beans-main/scripts/rfcx.py
python /media/dxy/Lexar/beans-main/scripts/hainan_gibbons.py
python /media/dxy/Lexar/beans-main/scripts/esc50.py
python /media/dxy/Lexar/beans-main/scripts/speech_commands.py
python /media/dxy/Lexar/beans-main/scripts/validate_data.py
