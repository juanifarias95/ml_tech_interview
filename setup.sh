#!/usr/bin/env bash

mkdir data
conda env create -f environment.yml
conda clean -y --all
