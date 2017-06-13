-include .build_release/Video.d

.build_release/Video.l: \
	.build_release/Video.d \

release/Video: .build_release/Video.l \
	.build_release/Video.o \

