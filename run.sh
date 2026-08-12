#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate nunifiw3

python -m iw3.gui
