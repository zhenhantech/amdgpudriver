# Makefile for Stream Priority Test
#
# 编译测试程序
# 使用: make all

CXX = hipcc
CXXFLAGS = -O2 -g

TARGETS = test_app_A test_app_B test_concurrent

all: $(TARGETS)

test_app_A: test_app_A.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test_app_B: test_app_B.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test_concurrent: test_concurrent.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
