data_path=data
save_path=csv_files
datasets="20ng rcv1"
rates="0.01 0.03 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
file_names="NB_EM_labeled.csv NB_EM_dataless.csv LR_SVM_labeled.csv LR_SVM_dataless.csv HierCost_labeled.csv HierCost_dataless.csv"

cnt=0
for dataset in $datasets
do
    mkdir -p ${save_path}/${dataset}
    for file_name in $file_names
    do
        echo cp ${data_path}/${dataset}/${file_name} ${save_path}/${dataset}/${file_name}
        cp ${data_path}/${dataset}/${file_name} ${save_path}/${dataset}/${file_name}
    done
    for rate in $rates
    do
        mkdir -p ${save_path}/${dataset}/${rate}
        for file_name in $file_names
        do
            echo cp ${data_path}/${dataset}/${rate}/${file_name} ${save_path}/${dataset}/${rate}/${file_name}
            cp ${data_path}/${dataset}/${rate}/${file_name} ${save_path}/${dataset}/${rate}/${file_name}
            # echo cp ${data_path}/${dataset}/${rate}/${file_name} ${save_path}/${file_name}
            # cp ${data_path}/${dataset}/${rate}/${file_name} ${save_path}/${file_name}
            (( cnt+=1 ))
        done
    done
done
