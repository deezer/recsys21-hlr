#!/bin/bash

#echo "python -m huitre train --verbose -p configs/echonest/10-core/transcf/fo_margin0.5.json"
#python -m huitre train \
#  --verbose \
#  -p configs/echonest/10-core/transcf/fo_margin0.5.json \
#  >& exp/logs/echonest_fo-10core/transcf/fo_lr0.0005_margin0.5.log
#
#echo "python -m huitre eval --verbose -p configs/echonest/10-core/transcf/fo_margin0.5.json"
#python -m huitre eval --verbose -p configs/echonest/10-core/transcf/fo_margin0.5.json

############################################################
#                       MOVIELENS
############################################################
#echo "python -m huitre train --verbose -p configs/mvlens/10-core/transcf/fo_margin0.5.json"
#python -m huitre train \
#  --verbose \
#  -p configs/mvlens/10-core/transcf/fo_margin0.5.json \
#  >& exp/logs/mvlens_fo-10core/transcf/fo_lr0.0005_margin0.5.log
#
#echo "python -m huitre eval --verbose -p configs/mvlens/10-core/transcf/fo_margin0.5.json"
#python -m huitre eval --verbose -p configs/mvlens/10-core/transcf/fo_margin0.5.json

############################################################
#                       AMAZON BOOK
############################################################
#echo "python -m huitre train --verbose -p configs/amzb/10-core/transcf/fo_margin0.5.json"
#python -m huitre train \
#  --verbose \
#  -p configs/amzb/10-core/transcf/fo_margin0.5.json \
#  >& exp/logs/amzb_fo-10core/transcf/fo_lr0.0005_margin0.5.log
#
#echo "python -m huitre eval --verbose -p configs/amzb/10-core/transcf/fo_margin0.5.json"
#python -m huitre eval --verbose -p configs/amzb/10-core/transcf/fo_margin0.5.json

############################################################
#                           YELP
############################################################
echo "python -m huitre train --verbose -p configs/yelp/5-core/transcf/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/yelp/5-core/transcf/fo_margin0.5.json \
  >& exp/logs/yelp_fo-5core/transcf/fo_lr0.0002_margin0.5.log

echo "python -m huitre eval --verbose -p configs/yelp/5-core/transcf/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/yelp/5-core/transcf/fo_margin0.5.json
