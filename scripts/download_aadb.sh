mkdir -p data/aadb/

# download labels
gdown --id 0BxeylfSgpk1MZ0hWWkoxb2hMU3c
unzip imgListFiles_label.zip
rm imgListFiles_label.zip
mv imgListFiles_label data/aadb/labels

# download preprocessed images
gdown --id 0BxeylfSgpk1MU2RsVXo3bEJWM2c
unzip datasetImages_warp256.zip
rm datasetImages_warp256.zip
mv datasetImages_warp256/ data/aadb/images