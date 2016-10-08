CPP=g++
CFLAG=-Wall -Wextra #-Werror -O3 

all: compile exec clean

compile: dataReader.o MLP.o MLPTrainer.o 
	${CPP} -o MLP $^

%.o: %.cpp
	${CPP} -o $@ -c $< ${CFLAG}

exec:
	./MLP

clean:
	rm *.o MLP
