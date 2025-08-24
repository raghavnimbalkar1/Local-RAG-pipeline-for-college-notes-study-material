CXX = g++
CXXFLAGS = -std=c++17 -O2 -I./faiss -fopenmp
LDFLAGS = -L./faiss/build/faiss -lfaiss -lopenblas -fopenmp

all: faiss_search

faiss_search: search.cpp
	$(CXX) $(CXXFLAGS) search.cpp -o faiss_search $(LDFLAGS)

clean:
	rm -f faiss_search
