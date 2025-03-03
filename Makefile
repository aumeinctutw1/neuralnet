CXX = clang++
CXXFLAGS = -Wall -Wextra -std=c++20
CXXLIBS = # Add cross-platform libs here if needed

# macOS-specific flags
MAC_INCLUDES = -I/opt/homebrew/include
MAC_LIB_PATH = -L/opt/homebrew/lib
MAC_LIBS = 

APP_NAME = nn
SOURCE_DIR = source
BUILD_DIR = build

OS := $(shell uname)

ifeq ($(OS), Darwin)
    TARGET = build_mac
else
    TARGET = build
endif

all: $(TARGET)

dirs:
	mkdir -p $(BUILD_DIR)

# Find all .cpp files in source dir
SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)

# Convert .cpp files to .o files in build dir
OBJS = $(patsubst $(SOURCE_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# Generic build (non-macOS)
build: $(OBJS)
	$(CXX) $(CXXFLAGS) $(CXXLIBS) $^ -o $(APP_NAME)

# macOS build (uses macOS-specific flags)
build_mac: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(MAC_LIB_PATH) $(MAC_LIBS) -o $(APP_NAME)

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(MAC_INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(APP_NAME)