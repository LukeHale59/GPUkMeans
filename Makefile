#
#  Description: This makefile allows you to build the assignment by typing
#  'make'.  Note that typing "BITS=64 make" will build a 64-bit executable
#  instead of a 32-bit executable.
#
#  Note: It's worth understanding how this Makefile works
#

# This defaults bits to 32, but allows it to be overridden on the command
# line
BITS = 64

# Output directory
ODIR  = obj32Serial
tmp  := $(shell mkdir -p $(ODIR))

# Basic compiler configuration and flags
CXX      = nvcc
CXXFLAGS = -MMD -O3 -m$(BITS) -std=c++11
LDFLAGS	 = -m$(BITS) -lpthread -lrt
# The basenames of the c++ files that this program uses
CXXFILES = kmeans-gpu

# The executable we will build
TARGET = $(ODIR)/p2

# Create the .o names from the CXXFILES
OFILES = $(patsubst %, $(ODIR)/%.o, $(CXXFILES))

# Create .d files to store dependency information, so that we don't need to
# clean every time before running make
DFILES = $(patsubst %.o, %.d, $(OFILES))

# Default rule builds the executable
all: $(TARGET)

# clean up everything by clobbering the output folder
clean:
	@echo cleaning up...
	@rm -rf $(ODIR)

# build an .o file from a .cc file
$(ODIR)/%.o: %.cu
	@echo [CXX] $< "-->" $@
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

# Link rule for building the target from .o files
$(TARGET): $(OFILES)
	@echo [LD] $^ "-->" $@
	@$(CXX) -o $@ $^ $(LDFLAGS)

# Remember that 'all' and 'clean' aren't real targets
.PHONY: all clean

# Pull in all dependencies
-include $(DFILES)
