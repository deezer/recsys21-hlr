#!/bin/bash

# MOVIELENS
#echo "python -m huitre analyse --verbose -p configs/mvlens/10-core/lrml/fo_margin0.2.json"
#python -m huitre analyse --verbose -p configs/mvlens/10-core/lrml/fo_margin0.2.json

#echo "python -m huitre analyse --verbose -p configs/mvlens/10-core/hlre/fo_margin0.2.json"
#python -m huitre analyse --verbose -p configs/mvlens/10-core/hlre/fo_margin0.2.json

echo "python -m huitre analyse --verbose -p configs/mvlens/10-core/jpte/fo_margin0.2.json"
python -m huitre analyse --verbose -p configs/mvlens/10-core/jpte/fo_margin0.2.json

# AMAZON BOOKS
#echo "python -m huitre analyse --verbose -p configs/amzb/10-core/lrml/fo_margin0.5.json"
#python -m huitre analyse --verbose -p configs/amzb/10-core/lrml/fo_margin0.5.json
#
#echo "python -m huitre analyse --verbose -p configs/amzb/10-core/hlr-e/fo_margin0.5.json"
#python -m huitre analyse --verbose -p configs/amzb/10-core/hlr-e/fo_margin0.5.json
