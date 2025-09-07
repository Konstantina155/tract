USE_SYS_TIME ?= 0

ifeq ($(USE_SYS_TIME), 1)
    CFLAGS += -DUSE_SYS_TIME
    FEATURE_FLAG = --features "use_sys_time"
else
    FEATURE_FLAG =
endif

all: mobilenet

mobilenet: tract.h libtract.a mobilenet.c
	cc -g mobilenet.c -o mobilenet $(CFLAGS) -L. -I. -ltract -lm -lpthread -ldl

build: tract.h libtract.a mobilenet.c

libtract.a:
	cargo build -p tract-ffi --release $(FEATURE_FLAG)
	cp ../../target/release/libtract.a .

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .

clean:
	rm -f mobilenet libtract.a tract.h