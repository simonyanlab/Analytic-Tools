FLS = the_model.so \
      the_model.c \

DIRS= build

all :	
	python setup.py build_ext --inplace
clean : 
	rm -f $(FLS)
	rm -rf $(DIRS)
