# Makefile for Stream Queue Limits Test

CXX = hipcc
CXXFLAGS = -O2 -g
TARGET = test_multiple_streams
SOURCE = test_multiple_streams.cpp

.PHONY: all clean test run help

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "✓ Compilation successful"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf logs
	@echo "✓ Clean completed"

test: $(TARGET)
	@echo "Running all comparison tests..."
	./run_tests.sh -a

test-16: $(TARGET)
	@echo "Testing with 16 streams..."
	./run_tests.sh 16

test-32: $(TARGET)
	@echo "Testing with 32 streams..."
	./run_tests.sh 32

test-64: $(TARGET)
	@echo "Testing with 64 streams..."
	./run_tests.sh 64

run: $(TARGET)
	@echo "Running test program (use: make run ARGS='<num_streams>')"
	@echo "Example: make run ARGS='16'"
	@if [ -z "$(ARGS)" ]; then \
		echo "Error: Please specify ARGS='<num_streams>'"; \
		exit 1; \
	fi
	./$(TARGET) $(ARGS)

help:
	@echo "Stream Queue Limits Test Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make               - Compile the test program"
	@echo "  make all           - Same as 'make'"
	@echo "  make clean         - Clean build artifacts and logs"
	@echo "  make test          - Run all comparison tests (16, 32, 64 streams)"
	@echo "  make test-16       - Run test with 16 streams"
	@echo "  make test-32       - Run test with 32 streams"
	@echo "  make test-64       - Run test with 64 streams"
	@echo "  make run ARGS='N'  - Run test with N streams"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test          # Run all tests"
	@echo "  make test-32       # Test with 32 streams"
	@echo "  make run ARGS='16' # Run with 16 streams"
	@echo ""
