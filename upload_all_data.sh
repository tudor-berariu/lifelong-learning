
for file in $1*.pkl; do
    name=${file##*/}
    base=${name%.txt}
    python upload_to_elasticsearch.py "$file"
done
