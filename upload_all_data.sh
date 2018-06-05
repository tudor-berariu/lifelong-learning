
for file in ../lifelong_tmp_data/*.pkl; do
    name=${file##*/}
    base=${name%.txt}
    python utils/upload_to_elasticsearch.py "$file"
done
