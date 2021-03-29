CC=gcc
CCFLAGS=-Wall -std=c99
LDFLAGS=-lm
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)
TARGET=goodfeatures

all: debug

debug: CCFLAGS += -DDEBUG -g
debug: $(TARGET)

release: CCFLAGS += -O3
release: $(TARGET)

$(TARGET): $(OBJECTS) $(CXXOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c %.h
	$(CC) $(CCFLAGS) -c $<

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

clean:
	rm -f *.pgm *.o $(TARGET) 