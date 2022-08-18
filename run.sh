#!/bin/sh
cd /home/ubuntu/scrl
echo $PATH
nohup streamlit run app.py --server.port 8502
