FC = gfortran

TARGET = test_sparse_ir
OBJS = test_sparse_ir.o sparse_ir.o sparse_ir_io.o sparse_ir_preset.o
LDLIBS = -lblas -llapack
#LDFLAGS = -L/usr/lib/aarch64-linux-gnu
LDFLAGS = -L/opt/homebrew/Cellar/openblas/0.3.20/lib

.SUFFIXES: .f90

%.o: %.f90
	$(COMPILE.f) $(OUTPUT_OPTION) $<
	#$(COMPILE.f) $(OUTPUT_OPTION) $< -fbounds-check

%.mod: %.f90 %.o
	@:

$(TARGET): $(OBJS)
	$(LINK.f) $^ $(LDLIBS) $(LDFLAGS) -o $@

test_sparse_ir.o: sparse_ir.mod sparse_ir_io.mod sparse_ir_preset.mod

test: $(TARGET)
	./test_sparse_ir

clean:
	rm -f *.o *.mod
