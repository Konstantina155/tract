CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Werror -g -no-pie
LDFLAGS = -I/hdd/papafrkon/dAIEdgeServer/include -I./include -L/hdd/papafrkon/dAIEdgeServer/lib -lmbedtls -lmbedx509 -lmbedcrypto

all: cbc_final gcm

cbc_final: cbc/cbc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm: gcm_final gcm_first gcm_encr

gcm_final: gcm/gcm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm_first: gcm/gcm_first.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gcm_encr: gcm/gcm_encr.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cbc_final gcm_final gcm_first gcm_encr
