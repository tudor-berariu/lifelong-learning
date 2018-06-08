#!/bin/bash
# Will open a tunnel to ports that correspond to Kibana (5601) and Elasticsearch server (9200)

ssh -L 5601:127.0.0.1:5601 -L 9200:127.0.0.1:9200 tempuser@141.85.232.73 -N -v -v