Repair all found reporting.pkl files.
python repair_reporting_data.py ../lifelong_learning_results/1528190673_mnist_50_perm_ewc/ 

Run entire process to upload all found reporting.pkl files to server. (Report -> eData -> server)
python upload_to_elasticsearch.py ../lifelong_learning_results/1528190673_mnist_50_perm_ewc/  

Command for to find files that have raised and error -> delete them. Used for buggy pkl file.
grep 'ERROR' repair_reporting_data.out | cut -d" " -f 5 | xargs -- dirname | xargs -I '{}' rm '{}'/.__end '{}'/reporting.pkl