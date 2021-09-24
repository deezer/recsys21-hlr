#!/bin/bash

#########################################################################################
#                                    YELP
#########################################################################################
echo "python -m huitre train --verbose -p configs/yelp/5-core/attcml/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/yelp/5-core/attcml/fo_margin0.5.json \
  >& exp/logs/yelp_fo-5core/attcml/fo_lr0.0005_margin0.5_nprefs50.log

echo "python -m huitre eval --verbose -p configs/yelp/5-core/attcml/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/yelp/5-core/attcml/fo_margin0.5.json

#########################################################################################
#                                    ECHONEST
#########################################################################################
#echo "python -m huitre train --verbose -p configs/echonest/10-core/attcml/fo_margin0.5.json"
#python -m huitre train \
#  --verbose \
#  -p configs/echonest/10-core/attcml/fo_margin0.5.json \
#  >& exp/logs/echonest_fo-10core/attcml/fo_lr0.0005_margin0.5_nprefs50.log
#
#echo "python -m huitre eval --verbose -p configs/echonest/10-core/attcml/fo_margin0.5.json"
#python -m huitre eval --verbose -p configs/echonest/10-core/attcml/fo_margin0.5.json

#########################################################################################
#                                    MOVIELENS-20M
#########################################################################################
#echo "python -m huitre train --verbose -p configs/mvlens/10-core/attcml/fo_margin0.2.json"
#python -m huitre train \
#  --verbose \
#  -p configs/mvlens/10-core/attcml/fo_margin0.2.json \
#  >& exp/logs/mvlens_fo-10core/attcml/fo_lr0.0002_margin0.2_nprefs50.log
#
#echo "python -m huitre eval --verbose -p configs/mvlens/10-core/attcml/fo_margin0.2.json"
#python -m huitre eval --verbose -p configs/mvlens/10-core/attcml/fo_margin0.2.json

#########################################################################################
#                                    BOOKCROSS
#########################################################################################
#echo "python -m huitre train --verbose -p configs/bookcross/10-core/attcml/fo_margin0.5.json"
#python -m huitre train \
#  --verbose \
#  -p configs/bookcross/10-core/attcml/fo_margin0.5.json \
#  >& exp/logs/bookcross_fo-10core/attcml/fo_lr0.0005_margin0.5_nprefs50.log
#
#echo "python -m huitre eval --verbose -p configs/bookcross/10-core/attcml/fo_margin0.5.json"
#python -m huitre eval --verbose -p configs/bookcross/10-core/attcml/fo_margin0.5.json
