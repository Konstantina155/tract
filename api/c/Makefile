USE_SYS_TIME ?= 0
USE_MEMORY_ONLY ?= 0
NUM_TOKENS ?= 10
USE_PII_REGEX ?= 0
USE_PII_GUARDRAIL ?= 0

ifeq ($(USE_SYS_TIME), 1)
    CFLAGS += -DUSE_SYS_TIME
    FEATURE_FLAG = --features "use_sys_time"
else ifeq ($(USE_PII_REGEX), 1)
    FEATURE_FLAG = --features "use_pii_regex"
else ifeq ($(USE_PII_GUARDRAIL), 1)
    FEATURE_FLAG = --features "use_pii_guardrail"
endif
ifeq ($(USE_MEMORY_ONLY), 1)
	CFLAGS += -DUSE_MEMORY_ONLY
endif
CFLAGS += -DNUM_TOKENS=$(NUM_TOKENS)

all: mobilenet

build: tract.h libtract.a mobilenet.c

mobilenet: tract.h libtract.a mobilenet.c
	cc -g mobilenet.c -o mobilenet $(CFLAGS) -L. -I. -ltract -lm -lpthread -ldl

libtract.a:
	cargo build -p tract-ffi --release $(FEATURE_FLAG)
	cp ../../target/release/libtract.a .
	strip --strip-debug libtract.a
	strip --strip-unneeded libtract.a

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .

clean:
	rm -f mobilenet libtract.a tract.h
