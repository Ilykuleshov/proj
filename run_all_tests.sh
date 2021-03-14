#!/bin/bash
cd src
for confname in ../configs/*; do
    python3 main.py --yaml_path "$confname"
done