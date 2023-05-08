wget http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
unzip ExtraSensory.per_uuid_features_labels.zip -d ./extrasensory
cd extrasensory
find . -name '*.csv.gz' | xargs -P 0 -I @ gzip -d @
rm ../ExtraSensory.per_uuid_features_labels.zip

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip
unzip wisdm-dataset.zip -d ./
rm wisdm-dataset.zip