wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
tar xzvf slot_test_data_full.tar.gz

wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
tar xzvf slot_train_data_full.tar.gz

for file in `ls ./slot_train_data_full`
do
    echo $file
    filename=slot_train_data_full/${file}
    shuf ${filename} | split -a1 -d -l $(( $(wc -l <${filename}) * 90 / 100 )) - ${filename}"_split"
done

for file in `ls ./slot_test_data_full`
do
    echo $file
    filename=slot_test_data_full/${file}
    shuf $filename | split -a1 -d -l $(( $(wc -l <$filename) * 90 / 100 )) - $filename"_split"
done


mkdir test
mkdir train
mv slot_test_data_full/*_split1 test
mv slot_train_data_full/*_split1 test
mv slot_test_data_full/*_split0 train
mv slot_train_data_full/*_split0 train

for file in `ls ./train | shuf -n 10`
do
    echo $file
    cp train/$file train/$file"_copy"
done

rm -rf slot_test_data_full slot_train_data_full
mv test slot_test_data_full
mv train slot_train_data_full

echo " Done~"