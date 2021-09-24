#!/bin/bash

#######################################################################################
#                                    MOVIELENS 20M
#######################################################################################
echo "python -m huitre train --verbose -p configs/mvlens/10-core/lrml/fo_margin0.2.json"
python -m huitre train \
  --verbose \
  -p configs/mvlens/10-core/lrml/fo_margin0.2.json \
  >& exp/logs/mvlens_fo-10core/lrml/fo_lr0.0002_margin0.2.log

echo "python -m huitre eval --verbose -p configs/mvlens/10-core/lrml/fo_margin0.2.json"
python -m huitre eval --verbose -p configs/mvlens/10-core/lrml/fo_margin0.2.json

#######################################################################################
#                                    ECHONEST
#######################################################################################

echo "python -m huitre train --verbose -p configs/echonest/10-core/lrml/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/echonest/10-core/lrml/fo_margin0.5.json \
  >& exp/logs/echonest_fo-10core/lrml/fo_lr0.0005_margin0.5.log

echo "python -m huitre eval --verbose -p configs/echonest/10-core/lrml/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/echonest/10-core/lrml/fo_margin0.5.json

########################################################################################
#                                        YELP
########################################################################################
echo "python -m huitre train --verbose -p configs/yelp/5-core/lrml/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/yelp/5-core/lrml/fo_margin0.5.json \
  >& exp/logs/yelp_fo-5core/lrml/fo_lr0.00075_margin0.5_copyram+clipram.log

echo "python -m huitre eval --verbose -p configs/yelp/5-core/lrml/fo_margin0.5.json"
python -m huitre eval --verbose -p configs/yelp/5-core/lrml/fo_margin0.5.json

#######################################################################################
#                                    AMAZON BOOK
#######################################################################################
echo "python -m huitre train --verbose -p configs/amzb/10-core/lrml/fo_margin0.5.json"
python -m huitre train \
  --verbose \
  -p configs/amzb/10-core/lrml/fo_margin0.5.json \
  >& exp/logs/amzb_fo-10core/lrml/fo_lr0.0005_margin0.5.log

echo "python -m huitre analyse --verbose -p configs/amzb/10-core/lrml/fo_margin0.5.json"
python -m huitre analyse --verbose -p configs/amzb/10-core/lrml/fo_margin0.5.json
