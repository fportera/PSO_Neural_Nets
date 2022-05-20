all: cnn

cnn: cnn.cpp 
	g++ -o cnn cnn.cpp -O3

clean:
	rm cnn

