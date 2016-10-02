CPP=clang++
CFLAG=-Wall -Wextra -v #-Werror -O3 

all: compile exec clean

compile: neurone.o MLP.o 
	${CPP} -o MLP $^

%.o: %.cpp
	${CPP} -o $@ -c $< ${CFLAG}

exec:
	./MLP

clean:
	rm *.o MLP
