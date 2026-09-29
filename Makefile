NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17

TARGET := signal_batch
SOURCE := signal_batch.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET) execution_log.txt sample_processed.csv

run:
	./$(TARGET) --signals 512 --samples 4096 --radius 2
