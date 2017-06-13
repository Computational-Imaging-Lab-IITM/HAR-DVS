current_dir=${PWD##*/}
output_dir="../../features_extracted/${current_dir}/"

if [ ! -d "$output_dir" ]; then
	# output directory doesn't exist so create one
	mkdir "$output_dir";
	echo "Output directory doesn't exist. So created it.";
fi

for file in ./*.avi; do
	../../../../lib/dense_trajectory_release_v1.2/release/DenseTrack "${file##*/}" >  "${output_dir}dense_features_${file##*/}.txt";
	echo "Extracted features for video: ${file##*/}";
done
