wget http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
unzip ExtraSensory.per_uuid_features_labels.zip
find . -name '*.csv.gz' | xargs -P 0 -I @ gzip -d @
rm ExtraSensory.per_uuid_features_labels.zip