-include .build_release/DenseTrack.d

.build_release/DenseTrack.l: \
	.build_release/DenseTrack.d \

release/DenseTrack: .build_release/DenseTrack.l \
	.build_release/DenseTrack.o \

