#!/bin/bash

########################################################################################
#                                     MOVIELENS-20M
########################################################################################
echo "python -m huitre train --verbose -p configs/mvlens/10-core/hlre/fo_margin0.2.json"
python -m huitre train \
  --verbose \
  -p configs/mvlens/10-core/hlre/fo_margin0.2.json \
  >& exp/logs/mvlens_fo-10core/hlre/fo_lr0.0002_margin0.2_nrel20_nprefs50.log

echo "python -m huitre eval --verbose -p configs/mvlens/10-core/hlre/fo_margin0.2.json"
python -m huitre eval --verbose -p configs/mvlens/10-core/hlre/fo_margin0.2.json

########################################################################################
#                                     ECHONEST
########################################################################################
echo "python -m huitre train --verbose -p configs/echonest/10-core/hlre/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/echonest/10-core/hlre/fo_margin0.5.json \
  >& exp/logs/echonest_fo-10core/hlre/fo_lr0.0005_margin0.5_nprefs50.log

echo "python -m huitre eval --verbose -p configs/echonest/10-core/hlre/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/echonest/10-core/hlre/fo_margin0.5.json

########################################################################################
#                                     YELP
########################################################################################
echo "python -m huitre train --verbose -p configs/yelp/5-core/hlre/fo_margin0.75.json"
python -m huitre train \
  --verbose \
  -p configs/yelp/5-core/hlre/fo_margin0.75.json \
  >& exp/logs/yelp_fo-5core/hlre/fo_margin0.75.log

echo "python -m huitre eval --verbose -p configs/yelp/5-core/hlre/fo_margin0.75.json"
python -m huitre eval --verbose -p configs/yelp/5-core/hlre/fo_margin0.75.json

#########################################################################################
#                                    AMAZON BOOK
#########################################################################################
#
echo "python -m huitre train --verbose -p configs/amzb/10-core/hlr-e/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/amzb/10-core/hlr-e/fo_margin0.5.json \
  >& exp/logs/amzb_fo-10core/hlr-e/fo_lr0.0005_margin0.5_nprefs20_unpop.log

echo "python -m huitre eval --verbose -p configs/amzb/10-core/hlr-e/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/amzb/10-core/hlr-e/fo_margin0.5.json
